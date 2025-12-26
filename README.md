# MemR3 README

This repo contains **MemR3**, a memory reasoning and retrieval agent system for answering questions over long conversations. It supports two backends:
- RAG over chunked chat histories with embedding search and optional cross-encoder reranking.
- Zep graph search over ingested conversation memories.

![MemR3 Pipeline.](./images/pipeline.png)

*Figure 1*: **MemR3 Pipeline**. MemR3 transforms retrieval into a closed-loop process: a router dynamically switches between Retrieve, Reflect, and Answer nodes while a global evidence–gap tracker maintains what is known and what is still missing. This enables iterative query refinement, targeted retrieval, and early stopping, making MemR3 an autonomous, backend-agnostic retrieval controller.

## Prerequisites
- Install dependencies: `pip install -r requirements.txt`.
- Required environment variables: `OPENAI_API_KEY`, `MODEL` (optional `OPENAI_BASE_URL`).
- RAG mode requires `EMBEDDING_MODEL` (optional `MEMR3_RERANK_MODEL`, `MEMR3_RAG_CACHE_DIR`).
- Zep mode requires `ZEP_API_KEY`.

## Datasets
We evaluate on the LoCoMo 10 conversation set using the Google Drive repository of the mem0 team: [LOCOMO Dataset](https://drive.google.com/drive/folders/1L-cTjTm0ohMsitsHg4dijSPJtqNflwX-?usp=drive_link).  
- `dataset/locomo10.json` is the base dataset for MemR3.
- `dataset/locomo10_rag.json` is the chunking-friendly variant used by the RAG backend.

## Usage (RAG and Zep)
Use the provided scripts to reproduce experiments and evaluation.
```
bash example_rag.sh
```

```
bash example_zep.sh
```

Outputs are saved under `results/`, and `evals.py` plus `generate_scores.py` compute BLEU/F1/LLM-judge summaries.

## Visualizations
![MemR3 Motivation.](./images/motivation.png)
*Figure 2*: Illustration of three memory-usage paradigms. **Full-Context** overloads the LLM with all memories and answers incorrectly; **Retrieve-then-Answer** retrieves relevant snippets but still miscalculates. In contrast, **MemR3** iteratively retrieves and reflects using an evidence–gap tracker (Acts 0–3), refines the query about Buddy’s adoption date, and produces the correct answer (3 months).

## Directory

```
|-- README.md
|-- evals.py
|-- example_rag.sh
|-- example_zep.sh
|-- generate_scores.py
|-- job.sh
|-- prompts.py
|-- run_experiments.py
|-- dataset
    |-- locomo10.json
    |-- locomo10_rag.json
|-- images
    |-- pipeline.png
    |-- motivation.png
|-- metrics
    |-- llm_judge.py
    |-- utils.py
|-- rag_cache-4.1
|-- rag_cache-4o
|-- results
|-- src
    |-- memr3_base.py
    |-- memr3_rag.py
    |-- memr3_zep.py
    |-- utils.py
    |-- zep
        |-- zep_ingestion.py
```
