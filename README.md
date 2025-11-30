# VectorVitals / MediLens — Encrypted Multi-Modal Clinical RAG (CyborgDB Hackathon 2025)

## Summary
MediLens (team **VectorVitals**) is a privacy-first prototype using **CyborgDB** for encrypted vector search. It ingests de-identified EHR text, generates embeddings locally, encrypts embeddings client-side, and stores them in CyborgDB. Clinicians query the system to get safe, explainable responses produced by a local LLM using retrieved encrypted context.

## What's included
- Data anonymization & preprocessing (basic PHI masking)
- Local embedding generation (`sentence-transformers`)
- CyborgDB integration for encrypted vector storage & retrieval
- Retrieval + response generation via local LLM (Flan-T5)
- Benchmarking for latency and throughput
- MIT license

## Quickstart (local, no Docker)
1. Clone repo into VS Code.
2. `python -m venv .venv && .venv/Scripts/activate` (Windows) or `source .venv/bin/activate` (macOS/Linux)
3. `pip install -r requirements.txt`
4. Copy `.env.example` → `.env` and fill in CyborgDB settings (endpoint, index key).
5. Put or edit `data/sample_ehr.jsonl` (already contains sample lines).
6. Run demo ingest & query:
   - `python src/run_demo.py ingest`  # create index & upsert vectors
   - `python src/run_demo.py query --q "What is the recommended anticoagulation for this patient?"`

## Notes & Security
- Keep `CYBORG_INDEX_KEY_BASE64` offline and secret.
- This dataset must be de-identified before ingestion — sample script includes simple masking functions; use production-grade PHI scrubbing for real EHR.
- The code decrypts embeddings **only in memory** for retrieval/fusion as required.

## Benchmarking
- `python src/benchmarks.py` runs ingestion + N queries, producing median/mean latency.

## License
MIT (see LICENSE)
