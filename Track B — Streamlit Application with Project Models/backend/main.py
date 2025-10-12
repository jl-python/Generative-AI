
from fastapi import FastAPI, UploadFile, Form
import tempfile
from .pipeline import answer_query

app = FastAPI(title="Week 7 Graph-RAG Backend (NetworkX)")

@app.post("/answer")
async def get_answer(query: str = Form(...), file: UploadFile = None):
    if file is None:
        return {"error": "No file uploaded"}
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = answer_query(query, tmp_path)
    except Exception as e:
        return {"error": str(e)}
    return result
