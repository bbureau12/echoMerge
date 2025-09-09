# 🔮 EchoMerge
*Guardrails for Safe LLM Memory Compaction*

---

## 📘 Overview
**EchoMerge** is a memory compaction and summarization system for **long-running LLM agents**.  
It merges overlapping or redundant entries while avoiding *hallucinated merges* (e.g., Henry V vs. Théoden King).

The system applies a **layered guardrail pipeline** — similarity, entity overlap, temporal alignment, and optional **LLM arbitration** — to decide whether memories can be safely combined. EchoMerge supports both **paragraph merges** and **one-sentence fusions**.

---

## ✨ Features
- **Similarity-aware** — `sentence-transformers` embeddings with configurable cosine thresholds.
- **Entity-aware** — spaCy NER + proper-noun fallback; blocks merges with mismatched entities.
- **Temporal reasoning** — same-day / within-hours alignment via a lightweight date merger.
- **Fusion vs Merge**
  - *Fusion*: one sentence for tightly related factoids.
  - *Merge*: short paragraph that de-duplicates, preserves unique details.
- **LLM arbitration (optional)** — validate/compose with a local Hugging Face model or an HTTP service.
- **Pluggable & testable** — each guard (similarity, entities, temporal, LLM) is independently testable.

---

## 🛠 Tech Stack
- **Core**: Python 3.11, dataclasses, typing  
- **NLP**: spaCy (`en_core_web_sm`; `en_core_web_trf` optional), `dateparser`  
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)  
- **LLM backends**:
  - Local HF (default) via `services/llm_service.HfLlmService`
  - HTTP microservice (optional; same interface)
- **Vector store**: Chroma (optional)

---

## 📂 Layout
```
services/
  llm_service.py        # local HF backend (merge & fusion prompts)
  merge_guard.py        # guardrail logic (similarity, entities, temporal, LLM)
  memory_service.py     # vector store + conversation buffer (optional)
summarization/
  memory_test_harness.py  # demo / test harness
tools/
  chroma_testkit.py     # seed/reset/inspect + compactor runner (CLI)
```

---

## ⚙️ LLM Service (local HF backend)

EchoMerge ships a simple LLM wrapper with two prompt “contracts”:

- **Merge prompt** → one concise, non-redundant paragraph (keeps shared core + adds unique detail), target ≈ *N* tokens.  
- **Fusion prompt** → exactly one sentence that preserves distinct facts and normalizes repeated time expressions.

```python
# services/llm_service.py (core calls)
from services.llm_service import HfLlmService

svc = HfLlmService(model_name="lmsys/vicuna-7b-v1.5", four_bit=True)

# One-sentence fusion (tight facts):
svc.summarize(
    ["I went to the store today.", "After the store, I read a book."],
    target_tokens=48,
    mode="fusion",
)

# Paragraph merge (overlapping notes):
svc.summarize([note_a, note_b], target_tokens=220, mode="merge")
```

### Constructor options

| Arg | What it does | Typical |
|---|---|---|
| `model_name` | HF model id | `lmsys/vicuna-7b-v1.5` |
| `four_bit` | Enable 4-bit quant (needs CUDA + bitsandbytes) | `True` on 12 GB GPUs |
| `max_ctx` | Max prompt tokens (approx) | `2048` |
| `temperature`, `top_p` | Sampling controls | `0.7`, `0.95` |
| `repetition_penalty` | Light anti-loop | `1.05` |
| `seed` | Determinism for tests | `None` |

**Hardware tips**
- 7B models run well on a **12 GB GPU** with `--four-bit`.
- If bitsandbytes isn’t available on Windows, use **WSL** or pick a quantized model that doesn’t require bnb.

---

## 🧪 Smoke Test (LLM only)

Ensure your GPU/CPU setup works before wiring into the compactor:

```bash
# Install core deps
pip install -U transformers accelerate safetensors sentence-transformers spacy dateparser chromadb

# spaCy English model
python -m spacy download en_core_web_sm

# Optional: bitsandbytes for 4-bit (Linux/WSL recommended)
pip install bitsandbytes

# Run LLM service harness
python services/llm_service.py --model lmsys/vicuna-7b-v1.5 --four-bit --fusion-target 48 --merge-target 220
```

You should see device map info and two outputs:
- **Fusion test** → one sentence  
- **Merge test** → one short paragraph

---

## 🧩 Using the LLM inside EchoMerge

The compactor expects an object with:

```python
class LlmService(Protocol):
    def summarize(self, texts: List[str], target_tokens: int = 220, mode: Literal["merge","fusion"]) -> str: ...
```

**Wiring it up (example runner):**

```python
from services.llm_service import HfLlmService
from memory_compactor import MemoryCompactor, CompactorConfig

llm = HfLlmService(model_name="lmsys/vicuna-7b-v1.5", four_bit=True)

cfg = CompactorConfig(
    collection_name="astraea_memories_test",
    similarity_threshold=0.87,
    target_tokens_cluster_summary=220,
    # fusion knobs:
    fusion_when_cluster_len_le=3,
    fusion_max_chars=600,
    fusion_target_tokens=48,
)

compactor = MemoryCompactor(client, cfg, llm)
compactor.run()   # or .compact() / .run_once()
```

> **Dry run:** if your runner supports `--dry-run`, the compactor will **not** write merges/archives but will log what it *would* do. In your code, the dry-run subclass honors MergeGuard’s suggested `force_mode` and still tracks budget.

---

## 🧠 Guardrail Summary
- **Similarity:** cosine on MiniLM; different thresholds for factoids, entity-backed merges, and first-person notes.
- **Entities:** spaCy NER (`PERSON`/`ORG`/`GPE`/`LOC`/`EVENT`/`WORK_OF_ART`/`NORP`) → fallback to contiguous `PROPN` spans.
- **First-person MVP:** inject `"i"`/`"we"` when they’re **syntactic subjects** to enable safe same-day fusions.
- **Temporal:** align “today/this morning/yesterday …” into same-day/within-hours.
- **LLM arbitration:** optional final check; if “no”, the merge is denied.

---

## 🚦 Example CLI (Testkit)

Seed, inspect neighbors, and run the compactor over a local Chroma collection:

```bash
python tools/chroma_testkit.py \
  --path ./chroma_test \
  --name astraea_memories_test \
  --reset --seed full \
  --inspect --neighbors 5 \
  --compact --dry-run \
  --threshold 0.87 \
  --target-tokens 220 \
  --max-llm-calls 12 \
  --max-llm-tokens-out 6000 \
  --fusion-len-le 3 \
  --fusion-max-chars 600 \
  --fusion-target 48 \
  --backend hf \
  --model-name lmsys/vicuna-7b-v1.5 \
  --four-bit
```

You’ll see which clusters would fuse/merge, estimated token use, and any budget blocks.

---

## 🧷 Troubleshooting
- **Model won’t load / OOM**
  - Pass `--four-bit` and ensure `bitsandbytes` is installed (Linux/WSL).
  - Reduce `--ctx` (prompt context) and/or use a smaller model.
- **bitsandbytes on Windows**
  - Prefer **WSL**; otherwise use a CPU-only flow or GGUF/GPTQ variants (not covered here).
- **spaCy model errors**
  - Ensure `en_core_web_sm` is installed; try `en_core_web_trf` if your GPU allows.
- **Weird merges**
  - Enable dry-run and print `signals` from `MergeGuard.decide_pair` to understand gating decisions.

---

## 🚀 Roadmap
- Streamlit demo dashboard  
- Interval/time-range temporal reasoning  
- Structured audit logs (JSON) for every decision  
- Blog post: “Guardrails for LLM Memory Systems”

---

## 📜 License
MIT — open for learning, remixing, and extending.
