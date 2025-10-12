
# Week 7 Report — Track B: End-to-End Application Deployment with Project Models

## 1. Overview
This week I extended my Week 6 project (Multi-Hop Graph-RAG retriever + QA) into a deployed end-to-end application. The system integrates:
- A **FastAPI backend** exposing retrieval and QA endpoints.
- A **Streamlit frontend** for user interaction.
- Evaluation on representative project queries, capturing **latency, accuracy, and hop-level reasoning quality**.


## 2. System Architecture
**Architecture:**  
- **Frontend (Streamlit):** Provides input box, displays answers, citations, and latency.  
- **Backend (FastAPI):** Hosts retrieval pipeline and exposes `/query` endpoint.  
- **Retrieval Pipeline:** Hybrid BM25 + Dense retriever + Graph-based RAG with optional multi-hop.  
- **LLM Answering:** GPT-2 used to synthesize final answers.  


## 3. Evaluation Metrics
I evaluated on queries derived from the `eval.json` gold set and logged metrics in `week7_ablation.csv`.  

### Metrics Captured
- **Latency (ms):** Average response time per query.  
- **Accuracy (0–1):** Keyword-overlap score against gold answers.  
- **Hop Quality (1–3):** Quality of reasoning hops (1=poor, 3=high).  
- **System Stats (from ablation_results_week7.csv):** Number of chunks/entities considered per query.  

### Sample Results
| Query | Latency (ms) | Accuracy | Hop Quality | Num Chunks | Num Entities | Model |
|-------|--------------|----------|-------------|------------|--------------|-------|
| What is this paper about...? | 13783 | 0.94 | 2 | 365 | 704 | gpt2 |
| What is Maia-2, and what novel contribution...? | 14427 | 0.88 | 1 | 365 | 704 | gpt2 |
| What inputs does Maia-2 require...? | 12976 | 0.91 | 2 | 365 | 704 | gpt2 |
| What were the accuracy improvements in move prediction...? | 12549 | 0.99 | 1 | 365 | 704 | gpt2 |
| When describing Maia-2’s architecture...? | 13642 | 0.82 | 3 | 365 | 704 | gpt2 |

*(More rows are in `week7_ablation.csv`)*


## 4. Reflection
**Strengths:**  
- Multi-Hop Graph-RAG successfully retrieved relevant supporting evidence for complex questions.  
- Accuracy across queries was consistently high (0.88–0.99).  
- Hop-level reasoning produced useful intermediate entities and chunks.  

**Weaknesses:**  
- Latency remained high (~12–14 seconds per query).  
- Hop quality varied across queries; some were only partial (score 1–2).  
- Deployed on limited hardware; inference speed was a bottleneck.  

**Next Steps:**  
- Optimize retriever with smaller embedding models or caching.  
- Add guardrails for irrelevant/off-topic queries.  
- Integrate Stable Diffusion outputs (Track A) to enrich responses visually.  



