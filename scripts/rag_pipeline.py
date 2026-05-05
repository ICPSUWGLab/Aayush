import re
import json
import numpy as np
import faiss
import pdfplumber
import requests
from sentence_transformers import SentenceTransformer

# -------------------------------
# 🔹 Embedding Model
# -------------------------------
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------------
# 🔹 PDF PROCESSING (FONT-AWARE)
# -------------------------------
def extract_text_with_headings(pdf_path):
    data = []

    current_section = {"heading": "Document", "content": ""}
    last_font_size = None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(extra_attrs=["size"])

            for word in words:
                text = word["text"].strip()
                font_size = word.get("size", 0)

                text = re.sub(r"[^\x00-\x7F]+", "", text)

                if not text:
                    continue

                # detect heading
                if last_font_size is not None and font_size > last_font_size + 1:
                    if current_section["content"]:
                        data.append(current_section)

                    current_section = {
                        "heading": text,
                        "content": ""
                    }
                else:
                    current_section["content"] += text + " "

                last_font_size = font_size

    if current_section["content"]:
        data.append(current_section)

    return data


# -------------------------------
# 🔹 CHUNKING
# -------------------------------
def chunk_by_headings(data, max_chunk_size=500):
    chunks = []

    for section in data:
        heading = section["heading"]
        words = section["content"].split()

        for i in range(0, len(words), max_chunk_size):
            chunk = " ".join(words[i:i + max_chunk_size])
            chunks.append(f"{heading}\n{chunk}")

    return chunks


# -------------------------------
# 🔹 EMBEDDINGS
# -------------------------------
def generate_embeddings(chunks):
    embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True
    )
    return np.array(embeddings)


# -------------------------------
# 🔹 FAISS INDEX
# -------------------------------
def store_in_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def query_faiss(index, query, chunks, k=5):
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    distances, indices = index.search(
        np.array(query_embedding),
        k
    )

    return [chunks[i] for i in indices[0] if i < len(chunks)]


# -------------------------------
# 🔹 LLM CALL (HuggingFace Router)
# -------------------------------
def get_structured_answer(context_chunks, query, hf_api_key):
    API_URL = "https://router.huggingface.co/v1/chat/completions"

    context_text = "\n".join(
        [f"[{i}] {chunk}" for i, chunk in enumerate(context_chunks)]
    )

    prompt = f"""
You MUST return ONLY valid JSON.

Context:
{context_text}

Question:
{query}

Return format:
{{
    "answer": "...",
    "citations": [0,1],
    "claims": ["fact1", "fact2"]
}}
"""

    headers = {
        "Authorization": f"Bearer {hf_api_key}"
    }

    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct:novita",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload
    )

    try:
        return response.json()["choices"][0]["message"]["content"]
    except Exception:
        return "{}"


# -------------------------------
# 🔹 SAFE JSON PARSER
# -------------------------------
def safe_parse_json(text):
    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except Exception:
                data = {}
        else:
            data = {}

    return {
        "answer": data.get("answer", ""),
        "citations": data.get("citations", []),
        "claims": data.get("claims", [])
    }
