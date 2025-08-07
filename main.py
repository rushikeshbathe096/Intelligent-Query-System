from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import requests
import fitz  # PyMuPDF
import os
import uuid
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import textwrap
import nltk
from dotenv import load_dotenv

# Initial setup
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

app = FastAPI()

class QueryRequest(BaseModel):
    pdf_url: str
    questions: list[str]

class QueryResponse(BaseModel):
    answers: dict

def download_pdf_from_url(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        file_path = f"/tmp/{uuid.uuid4().hex}.pdf"
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")

def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = " ".join([page.get_text() for page in doc])
        doc.close()
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction error: {e}")

def chunk_text(text, chunk_size=300, overlap=50):
    sentences = sent_tokenize(text)
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) <= chunk_size:
            chunk += " " + sentence
        else:
            chunks.append(chunk.strip())
            chunk = sentence
    if chunk:
        chunks.append(chunk.strip())

    final_chunks = []
    for i in range(len(chunks)):
        start = max(0, i - 1)
        final_chunks.append(" ".join(chunks[start:i + 1]))
    return final_chunks

def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings, chunks

def get_top_k_chunks(question, index, chunks, embeddings, k=5):
    q_embedding = embedder.encode([question])
    D, I = index.search(np.array(q_embedding), k)
    return [chunks[i] for i in I[0]]

def generate_answer(context, question):
    prompt = textwrap.dedent(f"""
        Answer the following question using the provided context only.
        If the answer is not in the context, say "Not Found".

        Context:
        {context}

        Question:
        {question}
    """)
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error from Gemini API: {e}"

@app.post("/query", response_model=QueryResponse)
async def query_pdf(request: QueryRequest):
    file_path = download_pdf_from_url(request.pdf_url)
    full_text = extract_text_from_pdf(file_path)
    chunks = chunk_text(full_text)
    index, embeddings, chunk_texts = create_faiss_index(chunks)

    result = {}
    for question in request.questions:
        top_chunks = get_top_k_chunks(question, index, chunk_texts, embeddings)
        context = "\n".join(top_chunks)
        answer = generate_answer(context, question)
        result[question] = answer

    os.remove(file_path)
    return {"answers": result}
