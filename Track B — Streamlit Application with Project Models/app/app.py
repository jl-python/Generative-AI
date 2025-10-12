
import streamlit as st, requests, time

st.set_page_config(page_title="Week 7 Graph RAG", layout="wide")
st.title("📚 Week 7 — Graph / Multi-hop RAG App (Track B)")

backend_url = "http://localhost:8000"


uploaded_file = st.file_uploader(
    "📄 Upload a document (PDF, DOCX, XLSX, CSV)",
    type=["pdf", "docx", "xlsx", "csv"]
)
query = st.text_input("🔎 Ask a question about your file")

if uploaded_file and query and st.button("Run Query"):
    with st.spinner("Processing..."):
        t0 = time.time()
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        data = {"query": query}

        try:
            r = requests.post(f"{backend_url}/answer", data=data, files=files, timeout=300)
            
            if r.status_code != 200:
                st.error(f"❌ Backend error ({r.status_code}): {r.text}")
            else:
                res = r.json()

                # ✅ handle backend errors gracefully
                if "error" in res:
                    st.error(f"⚠️ Backend returned an error: {res['error']}")
                elif "answer" not in res:
                    st.error("⚠️ Unexpected response from backend. No 'answer' field found.")
                    st.json(res)  # show what the backend sent for debugging
                else:
                    # ✅ normal successful case
                    latency = res.get("latency_ms", "?")
                    total = int((time.time() - t0) * 1000)

                    st.subheader("💡 Answer")
                    st.write(res["answer"])
                    st.caption(f"Backend latency: {latency} ms Total: {total} ms")

                    with st.expander("📚 Citations / Sources"):
                        for c in res.get("citations", []):
                            st.markdown(f"- {c['title']}")

                    with st.expander("🧩 Reasoning / Hops"):
                        for h in res.get("hops", []):
                            st.markdown(f"- **{h['step']}** — {h['notes']}")
        except Exception as e:
            st.error(f"Request failed: {e}")
else:
    st.info("Upload a file and enter a query to start.")


st.markdown("---")
st.caption("🧠 Built for Week 7 Track B — NetworkX Graph RAG + Chroma Vector Store")


