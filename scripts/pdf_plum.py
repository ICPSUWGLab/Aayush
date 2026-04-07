import os
import re
import json
import faiss
import numpy as np
import requests
import pdfplumber
from sentence_transformers import SentenceTransformer

from rag_evaluation import (
    evaluate_rag_advanced,
    summarize_advanced
)

# -------------------------------
# 🔹 Embedding Model
# -------------------------------
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# -------------------------------
# 🔹 PDF PROCESSING (FONT-AWARE)
# -------------------------------
def extract_text_with_headings(pdf_path):
    data = []

    current_section = {"heading": "Document", "content": ""}
    last_font_size = None

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):

            words = page.extract_words(extra_attrs=["size"])

            for word in words:
                text = word["text"].strip()
                font_size = word["size"]

                text = re.sub(r"[^\x00-\x7F]+", "", text)

                if not text:
                    continue

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
# 🔹 EMBEDDINGS + FAISS
# -------------------------------
def generate_embeddings(chunks):
    return embedding_model.encode(chunks, normalize_embeddings=True)


def store_in_faiss(embeddings):
    embeddings = np.array(embeddings)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def query_faiss(index, query, chunks, k=5):
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]


# -------------------------------
# 🔹 STRUCTURED ANSWER
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

    headers = {"Authorization": f"Bearer {hf_api_key}"}

    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct:novita",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    try:
        return response.json()["choices"][0]["message"]["content"]
    except:
        return "{}"


# -------------------------------
# 🔹 SAFE JSON PARSING
# -------------------------------
def safe_parse_json(text):
    try:
        data = json.loads(text)
    except:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except:
                data = {}
        else:
            data = {}

    return {
        "answer": data.get("answer", ""),
        "citations": data.get("citation", []),
        "claims": data.get("claims", [])
    }


# -------------------------------
# 🔹 MAIN
# -------------------------------
def main():
    pdf_path = "../data/pdfs/intelligent_systems.pdf"
    hf_api_key = os.getenv("HF_API_TOKEN")

    if not hf_api_key:
        raise ValueError("Set HF_API_TOKEN environment variable")

    print("Processing PDF with pdfplumber...")
    data = extract_text_with_headings(pdf_path)

    chunks = chunk_by_headings(data)
    print(f"\nTotal chunks: {len(chunks)}")

    embeddings = generate_embeddings(chunks)
    index = store_in_faiss(embeddings)

    mode = input("\nChoose mode: [1] Chat  [2] Evaluate : ")

    if mode == "2":
        qa_pairs = [
            ("What is the course code and the title of the course?", " CS-3270: Intelligent Systems."),
            ("What is the instructor's name?", "Dr. Md Shirajum Munir."),
            ("When is the exam 3 for the course?", "Wednesday, May 06, 12:00-2:00 PM"),
            ("Where is the office located?", "Technology Learning Center (TLC) 2248, 210 W Georgia Dr, Carrollton, GA 30117"),
        ]
        

        results = evaluate_rag_advanced(
            index=index,
            chunks=chunks,
            qa_pairs=qa_pairs,
            hf_api_key=hf_api_key,
            query_fn=query_faiss
        )

        summarize_advanced(results)
        with open("evaluation_results.txt", "w") as f:
            for r in results:
                f.write(json.dumps(r, indent=2) + "\n\n")

    else:
        while True:
            query = input("\nEnter your question: ")

            if query.lower() in ["exit", "quit"]:
                break

            retrieved_chunks = query_faiss(index, query, chunks)

            raw_output = get_structured_answer(
                retrieved_chunks,
                query,
                hf_api_key
            )

            parsed = safe_parse_json(raw_output)

            print("\nAnswer:", parsed.get("answer", "⚠️ No answer"))


if __name__ == "__main__":
    main()
