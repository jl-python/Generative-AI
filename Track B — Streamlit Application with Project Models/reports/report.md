
# Week 7 – Track B Report

## Overview
This app extends the Week 6 Graph RAG Streamlit demo into a modular deployment with a FastAPI backend + Streamlit frontend.

## Architecture
- **Backend:** FastAPI serving `answer_query()` (NetworkX graph + Chroma vector retrieval)  
- **Frontend:** Streamlit UI uploading documents and sending queries to the backend  
- **Models:** SpaCy NER + TinyLlama for generation + CrossEncoder for reranking  
- **Files supported:** PDF, DOCX, XLSX, CSV  

## Evaluation
Run 5+ queries and save results in `reports/metrics.json` and `ablation_results_week7.csv`.  
Include screenshots of the UI and latency outputs for submission.

## Future Work
- Replace NetworkX with Neo4j for persistent graph storage  
- Add live chat and speech output capabilities  
- Extend evaluation to compare reranking strategies


