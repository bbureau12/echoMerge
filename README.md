# 🔮 EchoMerge  
*Guardrails for Safe LLM Memory Compaction*

---

## 📘 Overview
**EchoMerge** is a memory compaction and summarization system designed for **long-running LLM agents**.  
It merges overlapping or redundant entries while avoiding *hallucinated merges* (e.g., Henry V vs Théoden King).  

The system applies a **layered guardrail pipeline** — similarity, entity overlap, temporal alignment, and optional LLM arbitration — to decide whether two (or more) memories can be safely combined.

---

## ✨ Features
- **Similarity-aware**  
  Uses `sentence-transformers` embeddings with configurable cosine thresholds.  

- **Entity-aware**  
  Leverages **spaCy NER** and fallback heuristics to block merges with mismatched entities.  

- **Temporal reasoning**  
  Detects same-day or same-window events using **dateparser**.  

- **Fusion vs Merge**  
  - *Fusion*: One concise sentence for tightly related factoids.  
  - *Merge*: Longer unified summary for overlapping but distinct notes.  

- **LLM arbitration**  
  Optional validation by a Hugging Face or HTTP-backed model.  

- **Pluggable & testable**  
  Each guard (similarity, entities, temporal, LLM) is independently testable.  

---

## 🛠 Tech Stack
- **Core**: Python 3.11, dataclasses, typing  
- **NLP**: [spaCy](https://spacy.io/) (`en_core_web_sm`), [dateparser](https://dateparser.readthedocs.io/), [sentence-transformers](https://www.sbert.net/)  
- **LLM Backends**: [Hugging Face Transformers](https://huggingface.co/docs/transformers/index), Vicuna, LLaMA, or any HTTP API  
- **Vector Store**: [Chroma](https://www.trychroma.com/) with [LangChain](https://www.langchain.com/) (optional)  

---

## 📂 Project Layout
services/
llm_service.py # wraps HuggingFace/HTTP models
merge_guard.py # guardrail logic for safe merges
memory_service.py # vector store + conversation buffer
summarization/
memory_test_harness.py # demo + test harness

---

## 🚦 Example Use Cases
- **Conversational Agents**  
  Compact user memories for long-running dialogue without cross-contamination.  

- **Knowledge Bases**  
  Merge redundant notes while preventing spurious merges.  

- **Portfolio Project**  
  Demonstrates applied hybrid AI (embeddings + NER + temporal + LLM arbitration).  

---

## 🧪 Quick Test
```bash
python services/merge_guard.py
[Case 1] fusion allowed (same-day, first-person narrative)
[Case 2] denied (low similarity)
[Case 3] merge allowed (entity overlap, high similarity)

---

## 🚀 Roadmap
 Add Streamlit demo dashboard

 Expand temporal reasoning with interval detection

 Audit logs (JSON) for every merge decision

 Blog post: “Guardrails for LLM Memory Systems”

## 📜 License

MIT — open for learning, remixing, and extending.