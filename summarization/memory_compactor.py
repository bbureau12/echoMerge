"""
Echo Compactor

What it does
------------
• Scans a Chroma collection of memories (documents + metadata).
• Detects near-duplicates / highly-overlapping memories via cosine similarity.
• Optionally clusters related memories and merges them into a concise summary.
• Archives originals while preserving provenance links.
• Maintains a light CostGuard to keep LLM calls under budget.

Assumptions
-----------
• Each memory is a document with metadata that may include fields like:
  {
    "type": "memory",               # or "note", "event", etc.
    "importance": 0-1,               # optional float importance score
    "created_at": "2025-08-30T...",  # ISO string
    "last_used_at": "2025-08-31T...",# ISO string, when memory was read
    "uses": int,                      # access count
    "topic": "ethics",               # optional topic tag
    "source": "astraea",             # who created it
    "cost_class": "cheap|std|exp"    # optional for cost guard
  }
• Collection already exists in Chroma and either stores embeddings or uses an embedder
  configured on the server.

Notes
-----
• Summarization is delegated to an LLM you control (e.g., Astraea). Replace the
  placeholder `summarize_with_llm` stub with your local inference call.
• This is designed to be run as a maintenance job (cron/daemon) or on-demand.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from merge_guard import MergeGuard, MergeGuardConfig
from services.llm_service import LlmService

try:
    import chromadb
    from chromadb.api.types import Documents, Embeddings, IDs, Metadatas
except Exception:  # pragma: no cover
    chromadb = None  # type: ignore


# -------------------------
# Configuration
# -------------------------
@dataclass
class CompactorConfig:
    collection_name: str = "astraea_memories"
    # Similarity & clustering
    similarity_threshold: float = 0.86          # cosine sim for pair merge consideration
    max_cluster_size: int = 8                   # upper bound on docs per cluster
    min_cluster_size_for_merge: int = 2

    # Token/length controls (approximate words->tokens ~ 0.75 for English)
    max_tokens_per_memory: int = 800            # summarize single overlong memory
    target_tokens_cluster_summary: int = 250

    # Cost guard
    max_llm_calls: int = 20                     # hard cap per run
    max_llm_tokens_out: int = 8000              # rough budget for output tokens

    # Scoring weights for prioritization
    w_importance: float = 0.50
    w_recency: float = 0.30
    w_usage: float = 0.20

    # Time decay (days)
    recency_half_life_days: float = 14.0

    # Archival policy
    archive_originals: bool = True

    # Fusion controls (one-sentence union for tiny subclusters)
    fusion_when_cluster_len_le: int = 3
    fusion_max_chars: int = 600
    fusion_target_tokens: int = 48


# -------------------------
# Utilities
# -------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_to_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def safe_get(md: Dict[str, Any], key: str, default: Any = None) -> Any:
    return md.get(key, default) if isinstance(md, dict) else default


def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    # Normalize rows and do dot product
    if X.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    return np.clip(Xn @ Xn.T, -1.0, 1.0)


def chunked(iterable, n: int):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


# -------------------------
# Cost Guard
# -------------------------
@dataclass
class CostGuard:
    max_calls: int
    max_tokens_out: int
    calls: int = 0
    tokens_out: int = 0

    def allow(self, est_tokens_out: int) -> bool:
        if self.calls + 1 > self.max_calls:
            return False
        if self.tokens_out + est_tokens_out > self.max_tokens_out:
            return False
        return True

    def commit(self, used_tokens_out: int):
        self.calls += 1
        self.tokens_out += used_tokens_out


# -------------------------
# LLM Stub (replace with your llm call)
# -------------------------

def summarize_with_llm(texts: List[str], target_tokens: int, mode: Optional[str] = None) -> str:
    """Replace with your local inference call.
    We keep a super simple deterministic merger when LLM is not available."""
    if not texts:
        return ""

    # --- simple console logging ---
    print(f"[summarize_with_llm] Summarizing {len(texts)} texts (target={target_tokens} tokens, mode={mode})")
    for i, t in enumerate(texts, 1):
        preview = (t[:120] + "...") if len(t) > 120 else t
        preview = preview.replace("\n", " ")
        print(f"  Text {i} preview: {preview}")

    # Naive fallback: pick most informative lines up to an approximate token limit
    approx_tokens = 0
    lines: List[str] = []
    for t in texts:
        for ln in t.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            # 1 token ~ 4 chars rough heuristic
            ln_tokens = max(1, len(ln) // 4)
            if approx_tokens + ln_tokens > target_tokens:
                continue
            lines.append(ln)
            approx_tokens += ln_tokens
    if not lines:
        return texts[0][: target_tokens * 4]
    header = "Merged memory summary (auto):"
    return header + "\n- " + "\n- ".join(lines)


# -------------------------
# Core Compactor
# -------------------------
class MemoryCompactor:
    def __init__(self, client: "chromadb.Client", cfg: CompactorConfig, llm: LlmService):
        self.client = client
        self.cfg = cfg
        self.guard = CostGuard(cfg.max_llm_calls, cfg.max_llm_tokens_out)
        self.col = self.client.get_or_create_collection(name=cfg.collection_name)
        self.llm = llm

        # MergeGuard — tweak thresholds here if you like
        mg_cfg = MergeGuardConfig(
            require_entity_overlap=True,
            require_temporal_alignment_if_no_entities=True,
            require_temporal_alignment_for_fusion=True,
            temporal_within_hours=24,
            use_llm_validator=False,   # flip on if you wire an LLM for arbitration
        )
        self.merge_guard = MergeGuard(mg_cfg)

    def _fetch_all(self, batch: int = 1000) -> Tuple[IDs, Documents, Metadatas, Embeddings]:
        ids: IDs = []
        docs: Documents = []
        metas: Metadatas = []
        embs: Embeddings = []
        offset = 0
        while True:
            res = self.col.get(
                include=["documents", "metadatas", "embeddings"],
                limit=batch,
                where={
                    "$and": [
                        {"archived": {"$ne": True}},
                        {"merged":   {"$ne": True}},
                    ]
                },
                offset=offset,
            )
            n = len(res.get("ids", []))
            if n == 0:
                break
            ids.extend(res["ids"])            # type: ignore
            docs.extend(res["documents"])     # type: ignore
            metas.extend(res["metadatas"])    # type: ignore
            embs.extend(res["embeddings"])    # type: ignore
            offset += n
        # Convert embeddings to ndarray
        X = np.array(embs, dtype=np.float32) if embs else np.zeros((0, 0), dtype=np.float32)
        return ids, docs, metas, X

    def _priority_scores(self, metas: Metadatas) -> np.ndarray:
        now = datetime.now(timezone.utc)
        half_life = self.cfg.recency_half_life_days
        scores = []
        for md in metas:
            imp = float(safe_get(md, "importance", 0.3))
            uses = float(safe_get(md, "uses", 0))
            lu = _iso_to_dt(safe_get(md, "last_used_at")) or _iso_to_dt(safe_get(md, "created_at"))
            if lu is None:
                rec = 0.5  # neutral
            else:
                days = max(0.0, (now - lu).total_seconds() / 86400.0)
                # decay: recency score in [0,1], 1 if very recent
                rec = 0.5 * (2 ** (-days / half_life)) * 2
            s = self.cfg.w_importance * imp + self.cfg.w_recency * rec + self.cfg.w_usage * (1 - math.exp(-uses / 5.0))
            scores.append(s)
        return np.array(scores, dtype=np.float32)

    def _clusters_from_similarity(self, S: np.ndarray, threshold: float) -> List[List[int]]:
        if S.size == 0:
            return []
        n = S.shape[0]
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            for j in range(i + 1, n):
                if S[i, j] >= threshold:
                    union(i, j)

        clusters: Dict[int, List[int]] = {}
        for i in range(n):
            r = find(i)
            clusters.setdefault(r, []).append(i)

        # Clip cluster size
        clipped = []
        for cl in clusters.values():
            if len(cl) >= self.cfg.min_cluster_size_for_merge:
                clipped.append(cl[: self.cfg.max_cluster_size])
        return clipped

    def _should_summarize_single(self, doc: str) -> bool:
        approx_tokens = max(1, len(doc) // 4)
        return approx_tokens > self.cfg.max_tokens_per_memory

    def _archive_ids(self, ids: List[str]):
        # We mark as archived rather than deleting to keep provenance.
        metas = self.col.get(ids=ids, include=["metadatas"])  # type: ignore
        new_metas = []
        for md in metas.get("metadatas", []):  # type: ignore
            md = dict(md or {})
            md["archived_at"] = _now_iso()
            md["archived"] = True
            new_metas.append(md)
        self.col.update(ids=ids, metadatas=new_metas)  # type: ignore

    def _add_merged_memory(
        self,
        texts: List[str],
        source_ids: List[str],
        metas: List[Dict[str, Any]],
        force_mode: Optional[str] = None
    ):
        # Choose mode — guard suggestion wins if valid
        if force_mode in ("fusion", "merge"):
            mode = force_mode
        else:
            total_chars = sum(len(t) for t in texts)
            if len(texts) <= self.cfg.fusion_when_cluster_len_le and total_chars <= self.cfg.fusion_max_chars:
                mode = "fusion"
            else:
                mode = "merge"

        target = self.cfg.fusion_target_tokens if mode == "fusion" else self.cfg.target_tokens_cluster_summary

        merged_text = self.llm.summarize(texts, target, mode=mode)

        # Guard against budget
        est_out = max(1, len(merged_text) // 4)
        if not self.guard.allow(est_out):
            # try smaller budget (fallback to fusion on the first text)
            merged_text = self.llm.summarize(texts[:1], min(128, target), mode=("fusion" if mode == "merge" else mode))
            est_out = max(1, len(merged_text) // 4)
            if not self.guard.allow(est_out):
                return
        self.guard.commit(est_out)

        merged_id = f"merged::{uuid.uuid4()}"
        base = {
            "type": "memory",
            "merged": True,
            "merged_from": source_ids,
            "created_at": _now_iso(),
            "importance": float(np.mean([safe_get(m, "importance", 0.3) for m in metas]) if metas else 0.3),
            "uses": 0,
            "topic": self._majority_topic(metas),
            "provenance": [{"id": sid, "meta": m} for sid, m in zip(source_ids, metas)],
            "merge_mode": mode,  # record how we summarized
        }
        self.col.add(ids=[merged_id], documents=[merged_text], metadatas=[base])  # type: ignore

    def _subclusters_with_merge_guard(
        self,
        cluster_indices: List[int],
        docs: List[str],
        S_full: np.ndarray
    ) -> List[Tuple[List[int], Optional[str]]]:
        """
        Within one similarity cluster, keep only edges approved by MergeGuard.
        Return connected components of the approved-edge graph AND an optional
        suggested_mode ('fusion' or None) inferred from guard-approved edges.
        """
        if not cluster_indices:
            return []

        m = len(cluster_indices)
        parent = list(range(m))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # Track guard-approved edges' modes
        edge_mode: Dict[frozenset, Optional[str]] = {}

        for li in range(m):
            gi = cluster_indices[li]
            for lj in range(li + 1, m):
                gj = cluster_indices[lj]
                sim = float(S_full[gi, gj])  # precomputed in the same ordering
                verdict = self.merge_guard.decide_pair(
                    docs[gi],
                    docs[gj],
                    precomputed_sim=sim,
                )
                if verdict["allow_merge"]:
                    union(li, lj)
                    edge_mode[frozenset((li, lj))] = verdict.get("mode")

        # Collect components
        comps: Dict[int, List[int]] = {}
        for li in range(m):
            r = find(li)
            comps.setdefault(r, []).append(li)

        # Convert to global indices; compute suggested_mode per subcluster
        out: List[Tuple[List[int], Optional[str]]] = []
        for comp in comps.values():
            if len(comp) < 2:
                continue

            # Suggest fusion if any approved edge in this component preferred 'fusion'
            has_fusion_edge = any(
                edge_mode.get(frozenset((i, j))) == "fusion"
                for i in comp for j in comp if i < j
            )
            suggested: Optional[str] = None
            if has_fusion_edge:
                total_chars = sum(len(docs[cluster_indices[li]]) for li in comp)
                if (len(comp) <= self.cfg.fusion_when_cluster_len_le and
                    total_chars <= self.cfg.fusion_max_chars):
                    suggested = "fusion"

            out.append(([cluster_indices[li] for li in comp], suggested))

        return out

    @staticmethod
    def _majority_topic(metas: List[Dict[str, Any]]) -> str | None:
        counts: Dict[str, int] = {}
        for m in metas:
            t = safe_get(m, "topic")
            if t:
                counts[t] = counts.get(t, 0) + 1
        if not counts:
            return None
        return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]

    def compact(self) -> Dict[str, Any]:
        ids, docs, metas, X = self._fetch_all()

        # Build S_full from the embeddings we actually have
        S_full = cosine_sim_matrix(X) if X.size else np.zeros((0, 0), dtype=np.float32)

        total = len(ids)
        if total == 0:
            return {"total": 0, "clusters": 0, "merged": 0, "archived": 0, "singles_summarized": 0}

        # Prioritize worklist by score (optional)
        scores = self._priority_scores(metas)
        order = np.argsort(-scores)
        ids   = [ids[i] for i in order]
        docs  = [docs[i] for i in order]
        metas = [metas[i] for i in order]
        X     = X[order]
        # IMPORTANT: reorder S_full to match 'order'
        if S_full.size:
            S_full = S_full[np.ix_(order, order)]

        clusters = self._clusters_from_similarity(S_full, self.cfg.similarity_threshold)
        merged_ct = 0
        archived_ct = 0
        singles_ct = 0

        # Handle clusters (merge & archive)
        for cl in clusters:
            # refine with MergeGuard (returns list of (indices, suggested_mode))
            subcls = self._subclusters_with_merge_guard(cl, docs, S_full)

            for sub in subcls:
                sub_indices, suggested_mode = sub
                sub_ids  = [ids[i] for i in sub_indices]
                sub_docs = [docs[i] for i in sub_indices]
                sub_meta = [metas[i] for i in sub_indices]

                # merge & archive this subcluster; honor guard suggestion
                self._add_merged_memory(sub_docs, sub_ids, sub_meta, force_mode=suggested_mode)
                if self.cfg.archive_originals:
                    self._archive_ids(sub_ids)
                    archived_ct += len(sub_ids)
                merged_ct += 1

                if self.guard.calls >= self.cfg.max_llm_calls:
                    break

            if self.guard.calls >= self.cfg.max_llm_calls:
                break

        # Handle single overlong memories
        head = min(500, len(ids))
        for i in range(head):
            if self.guard.calls >= self.cfg.max_llm_calls:
                break
            if safe_get(metas[i], "archived", False):
                continue
            approx_tokens = max(1, len(docs[i]) // 4)
            if approx_tokens > self.cfg.max_tokens_per_memory:
                short = summarize_with_llm(
                    [docs[i]],
                    target_tokens=self.cfg.target_tokens_cluster_summary,
                    mode="merge"
                )
                est_out = max(1, len(short) // 4)
                if not self.guard.allow(est_out):
                    continue
                self.guard.commit(est_out)
                # Update in-place with summarized version; keep original as provenance field
                md = dict(metas[i] or {})
                md["summarized_at"] = _now_iso()
                md["original_excerpt"] = docs[i][:512]
                self.col.update(ids=[ids[i]], documents=[short], metadatas=[md])  # type: ignore
                singles_ct += 1

        return {
            "total": total,
            "clusters": len(clusters),
            "merged": merged_ct,
            "archived": archived_ct,
            "singles_summarized": singles_ct,
            "llm_calls": self.guard.calls,
            "llm_tokens_out": self.guard.tokens_out,
        }