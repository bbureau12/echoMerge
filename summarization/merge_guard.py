# services/merge_guard.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal
from merge_date import merge_date
import re
import string

# --- Hard-require spaCy (fail fast if unavailable) ---
import spacy  # noqa: F401
try:
    _NLP = spacy.load("en_core_web_sm")
except Exception as e:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' is required for MergeGuard. "
        "Install with: pip install spacy==3.8.7 && python -m spacy download en_core_web_sm"
    ) from e

# Optional embedder
try:
    import numpy as np
except Exception:
    np = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore


_ALLOWED_LABELS = {"PERSON", "ORG", "NORP", "GPE", "LOC", "EVENT", "WORK_OF_ART"}

_FIRST_PERSON_RE = re.compile(r"\b(i|me|my|mine|we|us|our|ours)\b", re.IGNORECASE)

Mode = Literal["merge", "fusion", "skip"]


@dataclass
class MergeGuardConfig:
    # Similarity thresholds
    min_sim_merge: float = 0.78
    min_sim_factoid_merge: float = 0.80
    min_sim_first_person: float = 0.55
    min_sim_relaxed_with_entity: float = 0.62  # used when we have entity evidence

    # Entity checks
    require_entity_overlap: bool = True
    block_if_primary_entities_differ: bool = True

    # Topic gating
    require_topic_match: bool = False

    # LLM arbitration
    use_llm_validator: bool = True
    llm_prompt_max_chars: int = 600

    # Fusion heuristic
    fusion_len_le: int = 3
    fusion_total_chars_le: int = 600

    # Temporal gates
    require_temporal_alignment_for_fusion: bool = True
    require_temporal_alignment_if_no_entities: bool = True
    temporal_within_hours: int = 24


class MergeGuard:
    """
    Testable guard that decides whether two notes should be merged,
    and picks mode ('merge' vs 'fusion') if allowed.
    """
    # --- in __init__ of MergeGuard, enforce spaCy presence ---
    def __init__(self, cfg: MergeGuardConfig, embedder: Optional[Any] = None, nlp: Optional[Any] = None, llm: Optional[Any] = None):
        self.cfg = cfg
        self.llm = llm
        self.embedder = embedder

        # If you still want embeddings, keep your SentenceTransformer init here…
        if self.embedder is None and SentenceTransformer is not None:
            try:
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self.embedder = None

        # Hard-require spaCy
        if nlp is not None:
            self.nlp = nlp
        else:
            if spacy is None:
                raise RuntimeError("spaCy is required for MergeGuard. Install with: pip install spacy && python -m spacy download en_core_web_sm")
            try:
                self.nlp = spacy.load("en_core_web_trf")
            except Exception as e:
                raise RuntimeError(
                    "spaCy model 'en_core_web_sm' is required. "
                    "Install with: python -m spacy download en_core_web_sm"
                ) from e

        # keep your capword regex only if you want a last-ditch fallback; otherwise remove it
        # self._capword_re = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")


    # ---------- Public API ----------

    def decide_pair(
        self,
        text_a: str,
        text_b: str,
        meta_a: Optional[Dict[str, Any]] = None,
        meta_b: Optional[Dict[str, Any]] = None,
        precomputed_sim: Optional[float] = None,
    ) -> Dict[str, Any]:
        signals: Dict[str, Any] = {}

        # 0) Topic gate
        if self.cfg.require_topic_match:
            ta = (meta_a or {}).get("topic")
            tb = (meta_b or {}).get("topic")
            signals["topic_a"], signals["topic_b"] = ta, tb
            if ta and tb and ta != tb:
                return self._deny("topic_mismatch", signals)

        # 1) Semantic similarity (or LLM fallback if missing)
        sim = precomputed_sim if precomputed_sim is not None else self._similarity(text_a, text_b)
        signals["cosine_sim"] = float(sim) if sim is not None else None
        if sim is None:
            return self._llm_fallback(text_a, text_b, "no_similarity_available", signals)

        # 1.5) Temporal proximity
        temporal = merge_date(text_a, text_b, hours_window=self.cfg.temporal_within_hours)
        same_day, within_hours = temporal.same_day, temporal.within_hours
        temporal_tight = bool(same_day or within_hours)
        signals["temporal"] = {
            "same_day": same_day,
            "within_hours": within_hours,
            "best_pair": tuple(map(str, temporal.best_pair)) if temporal.best_pair else None,
        }

        # 2) Entities once (no fallbacks)
        ents_a = self._entities(text_a)
        ents_b = self._entities(text_b)
        signals["entities_a"], signals["entities_b"] = ents_a, ents_b
        has_overlap = bool(ents_a and ents_b and ents_a.intersection(ents_b))

        pa = self._primary_entity(ents_a)
        pb = self._primary_entity(ents_b)
        signals["primary_a"], signals["primary_b"] = pa, pb

        # 2.1) First-person + temporal lane (single place, before normal sim gate)
        if self._first_person_exception(text_a, text_b):
            if self.cfg.require_temporal_alignment_for_fusion and not temporal_tight:
                signals["first_person_lane"] = "blocked_no_temporal_alignment"
            else:
                if sim >= self.cfg.min_sim_first_person:
                    signals["first_person_lane"] = "allowed"
                    signals["proposed_mode"] = "fusion"
                    return self._allow("fusion", "first_person_fusion_exception", signals)
                else:
                    signals["first_person_lane"] = (
                        f"sim_below_first_person_min({sim:.3f} < {self.cfg.min_sim_first_person})"
                    )

        # 3) Normal similarity gate (with relaxation on entity evidence)
        is_factoid = self._is_factoid(text_a) and self._is_factoid(text_b)
        min_req = self.cfg.min_sim_factoid_merge if is_factoid else self.cfg.min_sim_merge
        print("DEBUG ents_a:", ents_a, "ents_b:", ents_b, "overlap:", has_overlap)
        if has_overlap or (pa and pb and pa == pb):
            # Relax the bar when we have strong entity evidence
            min_req = min(min_req, self.cfg.min_sim_relaxed_with_entity)

        if sim < min_req:
            return self._deny(f"similarity_below_threshold({sim:.3f} < {min_req})", signals)

        # 4) Entity overlap + primary mismatch gating (with generic-event temporal bypass)
        if self.cfg.require_entity_overlap:
            if not has_overlap:
                if self.cfg.require_temporal_alignment_if_no_entities:
                    if not temporal_tight:
                        return self._deny("no_entities_and_no_temporal_alignment", signals)
                else:
                    return self._deny("no_entity_overlap", signals)

        if self.cfg.block_if_primary_entities_differ and pa and pb and pa != pb:
            # Bypass if both are generic events AND time is tight
            if temporal_tight and self._is_generic_event(text_a) and self._is_generic_event(text_b):
                signals["primary_bypass"] = "generic_event_temporal_tight"
            else:
                return self._deny("primary_entities_differ", signals)

        # 5) Decide mode + optional LLM validator
        mode = self._propose_mode([text_a, text_b])
        signals["proposed_mode"] = mode

        if self.cfg.use_llm_validator and self.llm is not None:
            verdict = self._llm_same_event_check(text_a, text_b)
            signals["llm_verdict"] = verdict
            if verdict == "no":
                return self._deny("llm_disagrees_same_event", signals)

        return self._allow(mode, "passed_all_guards", signals)

    # ---------- Internal helpers ----------

    def _similarity(self, a: str, b: str) -> Optional[float]:
        if self.embedder is None or np is None:
            return None
        try:
            X = self.embedder.encode([a, b], normalize_embeddings=True)
            return float(np.dot(X[0], X[1]))
        except Exception:
            return None

    @staticmethod
    def _is_factoid(t: str) -> bool:
        t = t.strip()
        sentences = re.split(r"[.!?]+", t)
        words = len(t.split())
        return (words <= 25) and (sum(1 for s in sentences if s.strip()) <= 1)

    def _llm_fallback(self, a: str, b: str, reason: str, signals: Dict[str, Any]) -> Dict[str, Any]:
        if self.cfg.use_llm_validator and self.llm is not None:
            verdict = self._llm_same_event_check(a, b)
            signals["llm_verdict"] = verdict
            if verdict == "yes":
                signals["reason_no_sim"] = reason
                return self._allow("merge", "llm_allowed_without_similarity", signals)
        return self._deny(reason, signals)

    # --- replace _entities with spaCy-only version ---
    def _entities(self, t: str) -> set[str]:
        """
        Return normalized entity strings.
        1) Prefer spaCy NER on trusted labels.
        2) If NER is empty/weak, fall back to contiguous PROPN spans (proper nouns).
        """
        if self.nlp is None:
            raise RuntimeError("spaCy pipeline not available")

        doc = self.nlp(t)
        ents: set[str] = set()

        # --- 1) NER-first ---
        for ent in doc.ents:
            if ent.label_ in _ALLOWED_LABELS:
                s = " ".join(ent.text.strip().split()).lower()
                if s:
                    ents.add(s)

        # If NER found something meaningful, use it
        if len(ents) >= 1:
            return ents

        # --- 2) PROPN fallback: build contiguous proper-noun spans ---
        # Example: "Chapter 1 (Loomings): Ishmael, uneasy ashore..."
        # -> "loomings", "ishmael"
        spans: list[str] = []
        i = 0
        while i < len(doc):
            tok = doc[i]
            if tok.pos_ == "PROPN":
                j = i + 1
                while j < len(doc) and doc[j].pos_ == "PROPN":
                    j += 1
                span = doc[i:j]
                s = " ".join(span.text.strip().split()).lower()
                # very light filtering: skip 1-char and numeric-ish bits
                if len(s) > 1 and not any(ch.isdigit() for ch in s):
                    spans.append(s)
                i = j
            else:
                i += 1

        return set(spans)



    def _is_first_person(self, t: str) -> bool:
        return bool(_FIRST_PERSON_RE.search(t))

    def _first_person_exception(self, a: str, b: str) -> bool:
        # short, factual, and first-person on both sides
        return self._is_factoid(a) and self._is_factoid(b) and self._is_first_person(a) and self._is_first_person(b)

    # --- optional labeled primary-entity helper ---
    def _primary_entity(self, ents: set[str], labeled: Optional[Dict[str, str]] = None) -> Optional[str]:
        if not ents:
            return None
        if not labeled:
            # if you want labels, build them on the fly with spaCy:
            try:
                doc = self.nlp(" ".join(ents))
                labmap: Dict[str, str] = {}
                for ent in doc.ents:
                    if ent.text.lower() in ents:
                        labmap[ent.text.lower()] = ent.label_
                labeled = labmap
            except Exception:
                labeled = {}
        # scoring: PERSON > EVENT > ORG/GPE/LOC, then by length
        priority = {"PERSON": 4, "EVENT": 3, "ORG": 2, "GPE": 2, "LOC": 2}
        def score(e: str) -> tuple[int, int]:
            lab = labeled.get(e, "")
            return (priority.get(lab, 0), len(e))
        return sorted(ents, key=score, reverse=True)[0]


    # --- spaCy-only generic event detector ---
    def _is_generic_event(self, text: str) -> bool:
        """
        True if the text clearly describes a generic calendar-like event:
        - has an EVENT entity; or
        - has a root/head that is an activity/meeting noun (via POS/dep), judged heuristically.
        """
        if self.nlp is None:
            raise RuntimeError("spaCy pipeline not available")
        doc = self.nlp(text)

        # Strong signal: EVENT entities present
        if any(ent.label_ == "EVENT" for ent in doc.ents):
            return True

        # Heuristic: root is a NOUN/PROPN that *looks* like a schedulable activity
        # (No keywords; rely on morphology + typical “meetingy” structure)
        roots = [tok for tok in doc if tok.dep_ == "ROOT"]
        if roots and roots[0].pos_ in {"NOUN", "PROPN"}:
            return True

        # Noun chunks with nominal heads as main predicate often indicate events/meetings
        for nc in doc.noun_chunks:
            if nc.root.dep_ in {"ROOT", "attr", "dobj", "pobj"} and nc.root.pos_ in {"NOUN", "PROPN"}:
                # If text also has DATE/TIME entities, even better
                if any(e.label_ in {"DATE", "TIME"} for e in doc.ents):
                    return True

        return False


    def _propose_mode(self, texts: List[str]) -> Mode:
        total_chars = sum(len(t) for t in texts)
        if len(texts) <= self.cfg.fusion_len_le and total_chars <= self.cfg.fusion_total_chars_le:
            return "fusion"
        return "merge"

    def _llm_same_event_check(self, a: str, b: str) -> Literal["yes", "no", "unknown"]:
        msg = (
            "Answer strictly 'yes' or 'no'.\n"
            "Question: Do the following two notes describe the SAME real-world event/fact?\n\n"
            f"Note A: {a[:self.cfg.llm_prompt_max_chars]}\n"
            f"Note B: {b[:self.cfg.llm_prompt_max_chars]}\n\n"
            "Answer:"
        )
        try:
            out = self.llm.summarize([msg], target_tokens=8, mode="fusion").lower()
            if "yes" in out and "no" not in out:
                return "yes"
            if "no" in out and "yes" not in out:
                return "no"
            return "unknown"
        except Exception:
            return "unknown"

    @staticmethod
    def _allow(mode: Mode, reason: str, signals: Dict[str, Any]) -> Dict[str, Any]:
        return {"allow_merge": True, "mode": mode, "reason": reason, "signals": signals}

    @staticmethod
    def _deny(reason: str, signals: Dict[str, Any]) -> Dict[str, Any]:
        return {"allow_merge": False, "mode": "skip", "reason": reason, "signals": signals}