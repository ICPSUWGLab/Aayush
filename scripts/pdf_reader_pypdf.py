import os
import re
import json
import faiss
import numpy as np
import requests
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

from rag_evaluation import (
    evaluate_rag_advanced,
    summarize_advanced
)

# -------------------------------
# 🔹 Embedding Model
# -------------------------------
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


 #-------------------------------
 #🔹 PDF Processing
 #-------------------------------
def extract_text_with_headings(pdf_path):
    reader = PdfReader(pdf_path)
    data = []
    current_section = {"heading": "Document", "content": ""}

    for page in reader.pages:
        text = page.extract_text()
        if not text:
            continue

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.isupper() and len(line.split()) < 10:
                if current_section["content"]:
                    data.append(current_section)
                current_section = {"heading": line, "content": ""}
            else:
                current_section["content"] += line + " "

    if current_section["content"]:
        data.append(current_section)

    return data

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
# 🔹 Embeddings + FAISS
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
# 🔹 Structured Answer (CRITICAL)
# -------------------------------
def get_structured_answer(context_chunks, query, hf_api_key):
    API_URL = "https://router.huggingface.co/v1/chat/completions"

    context_text = "\n".join(
        [f"[{i}] {chunk}" for i, chunk in enumerate(context_chunks)]
    )

    prompt = f"""
    Use ONLY the provided context.

    Context:
    {context_text}

    Question:
    {query}

    Instructions:
    - Answer the question
    - Cite chunk numbers used
    - Extract factual claims
    - If unsure, say "I don't know"

    Output STRICT JSON:
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
# 🔹 Safe JSON Parsing
# -------------------------------
def safe_parse_json(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    # ✅ enforce schema
        return {
            "answer": "",
            "citations": [],
            "claims": []
        }

# -------------------------------
# 🔹 MAIN
# -------------------------------
def main():
    pdf_path = "../data/pdfs/intelligent_systems.pdf"
    hf_api_key = os.getenv("HF_API_TOKEN")

    if not hf_api_key:
        raise ValueError("Set HF_API_KEY as environment variable")

    print("Processing PDF...")
    data = extract_text_with_headings(pdf_path)
    chunks = chunk_by_headings(data)

    print(f"Total chunks: {len(chunks)}")

    embeddings = generate_embeddings(chunks)
    index = store_in_faiss(embeddings)

    mode = input("\nChoose mode: [1] Chat  [2] Evaluate : ")

    # ---------------------------
    # 🔹 EVALUATION MODE
    # ---------------------------
    if mode == "2":
        qa_pairs = [
            ("What is the course code?", "Intelligent Systems."),
            ("What is westga.edu email?", "Dr. Md Shirajum Munir."),
            ("When is the exam 3 for the course?", "May 06, 12:00-2:00 PM"),
            ("Where is the office located?", "Technology Learning Center (TLC) 2248, 210 W Georgia Dr, Carrollton, GA 30117"),
            ("What is the description of Intelligent System course?", "Application and survey of problem-solving methods in artificial intelligence with emphasis on heuristic programming, production systems, neural networks, agents, social implications of computing, and professional ethics and responsibilities."),
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

    # ---------------------------
    # 🔹 CHAT MODE
    # ---------------------------
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

            print("\nAnswer:", parsed["answer"])
            print("Citations:", parsed["citations"])


if __name__ == "__main__":
    main()
