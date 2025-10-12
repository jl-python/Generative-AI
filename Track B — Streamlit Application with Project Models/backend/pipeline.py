"""
pipeline.py — Week 7 Track B
Implements Graph RAG using NetworkX (in-memory graph) + Chroma vector store.
Supports PDF, DOCX, XLSX, CSV uploads.
"""

import os, re, io, time, tempfile, networkx as nx, spacy, pandas as pd, docx
from typing import List
from sentence_transformers import CrossEncoder
from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ---------- Model & Embeddings ----------
def load_spacy(use_trf=False):
    try:
        return spacy.load("en_core_web_trf") if use_trf else spacy.load("en_core_web_sm")
    except:
        return spacy.load("en_core_web_sm")

nlp = load_spacy()
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
generator = hf_pipeline(
    "text-generation",
    model=AutoModelForCausalLM.from_pretrained("gpt2"),
    tokenizer=AutoTokenizer.from_pretrained("gpt2"),
    max_new_tokens=256,
    do_sample=True,
    temperature=0.3
)

# ---------- File Loaders ----------
def load_file_to_texts(path: str):
    ext = os.path.splitext(path)[1].lower()
    texts = []

    if ext == ".pdf":
        loader = PyPDFLoader(path)
        docs = loader.load()
        texts = [d.page_content for d in docs]
    elif ext in [".docx", ".doc"]:
        doc = docx.Document(path)
        texts = [p.text for p in doc.paragraphs if p.text.strip()]
    elif ext == ".csv":
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            texts.append(" | ".join([f"{c}: {row[c]}" for c in df.columns]))
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
        for _, row in df.iterrows():
            texts.append(" | ".join([f"{c}: {row[c]}" for c in df.columns]))
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.create_documents(texts)
    return [c.page_content for c in chunks], chunks

# ---------- Graph RAG Logic ----------
def build_weighted_entity_graph(texts: List[str]) -> nx.Graph:
    G = nx.Graph()
    for i, text in enumerate(texts):
        doc = nlp(text)
        ents = [ent.text for ent in doc.ents]
        for e1 in ents:
            for e2 in ents:
                if e1 != e2:
                    if G.has_edge(e1, e2):
                        G[e1][e2]["weight"] += 1
                    else:
                        G.add_edge(e1, e2, weight=1, chunk=i)
    return G

def decompose(query: str) -> List[str]:
    parts = re.split(r"\band\b|,|;|\bthen\b", query, flags=re.I)
    return [p.strip() for p in parts if p.strip()]

def generate_answer(query: str, context: str) -> str:
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    out = generator(prompt)
    text = out[0]["generated_text"]
    return text.split("Answer:", 1)[-1].strip() if "Answer:" in text else text.strip()



def graph_rag(query: str, texts: List[str], vectordb: Chroma, G: nx.Graph, top_k: int = 5):
    doc = nlp(query)
    ents = [ent.text for ent in doc.ents]
    related = set()
    for e in ents:
        if e in G:
            neighbors = sorted(G.neighbors(e), key=lambda n: G[e][n]["weight"], reverse=True)
            related.update(neighbors[:3])
    dense_docs = vectordb.similarity_search(query, k=top_k * 2)
    filtered = [d for d in dense_docs if any(r in d.page_content for r in related)] or dense_docs
    return filtered[:top_k], list(related)

# ---------- Main Entry ----------
def answer_query(query: str, file_path: str):
    t0 = time.time()
    texts, chunks = load_file_to_texts(file_path)
    vectordb = Chroma.from_documents(chunks, embedding=emb, collection_name="tmp_collection")
    G = build_weighted_entity_graph(texts)
    docs, entities = graph_rag(query, texts, vectordb, G)
    context = "\n\n".join([d.page_content for d in docs])
    answer = generate_answer(query, context)

    latency_ms = int((time.time() - t0) * 1000)
    num_chunks = len(chunks)
    num_entities = len(G.nodes())

    # ✅ Log metrics after computing
    log_metrics(latency_ms, query, num_chunks, num_entities)

    
    return {
    "answer": answer,
    "citations": [{"title": f"Chunk {i+1}", "url": "#"} for i, _ in enumerate(docs)],
    "hops": [{"step": f"Hop {i+1}", "notes": s} for i, s in enumerate(decompose(query))],
    "latency_ms": latency_ms
    }


# ---------- Metrics Logging ----------
import json, csv, os

def log_metrics(latency_ms, query, num_chunks, num_entities, model_name="gpt2"):
    """
    Save metrics for each query to both JSON and CSV files inside /reports.
    """
    metrics_path = os.path.join(os.path.dirname(__file__), "../reports/metrics.json")
    ablation_path = os.path.join(os.path.dirname(__file__), "../reports/ablation_results_week7.csv")

    # make sure the reports folder exists
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    entry = {
        "query": query,
        "latency_ms": latency_ms,
        "num_chunks": num_chunks,
        "num_entities": num_entities,
        "model": model_name
    }

    # --- JSON ---
    try:
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                data = json.load(f)
        else:
            data = []
        data.append(entry)
        with open(metrics_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print("⚠️ metrics.json logging failed:", e)

    # --- CSV ---
    try:
        write_header = not os.path.exists(ablation_path)
        with open(ablation_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["query", "latency_ms", "num_chunks", "num_entities", "model"])
            writer.writerow([query, latency_ms, num_chunks, num_entities, model_name])
    except Exception as e:
        print("⚠️ ablation_results_week7.csv logging failed:", e)
