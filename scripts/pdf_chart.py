import os
import re
import json
import faiss
import numpy as np
import requests
import pdfplumber
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 🔹 Embedding Model
# -------------------------------
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# -------------------------------
# 🔹 PDF PROCESSING
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
                font_size = word["size"]

                text = re.sub(r"[^\x00-\x7F]+", "", text)

                if not text:
                    continue

                if last_font_size and font_size > last_font_size + 1:
                    if current_section["content"]:
                        data.append(current_section)

                    current_section = {"heading": text, "content": ""}
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
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def query_faiss(index, query, chunks, k=5):
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]


# -------------------------------
# 🔹 LLM ANSWER
# -------------------------------
def get_structured_answer(context_chunks, query, hf_api_key):
    API_URL = "https://router.huggingface.co/v1/chat/completions"

    context_text = "\n".join(
        [f"[{i}] {chunk}" for i, chunk in enumerate(context_chunks)]
    )

    prompt = f"""
Return ONLY valid JSON.

Context:
{context_text}

Question:
{query}

Format:
{{
    "answer": "...",
    "citations": [0,1],
    "claims": ["fact1"]
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
# 🔹 SAFE JSON
# -------------------------------
def safe_parse_json(text):
    try:
        data = json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(match.group()) if match else {}

    return {
        "answer": data.get("answer", ""),
        "citations": data.get("citations", []),  # ✅ fixed bug
        "claims": data.get("claims", [])
    }


# -------------------------------
# 🔹 METRICS
# -------------------------------
def exact_match(pred, truth):
    return int(pred.strip().lower() == truth.strip().lower())


def f1_score(pred, truth):
    p = pred.lower().split()
    t = truth.lower().split()

    common = set(p) & set(t)
    if not common:
        return 0

    precision = len(common) / len(p)
    recall = len(common) / len(t)

    return 2 * precision * recall / (precision + recall)


def similarity(pred, truth):
    emb1 = embedding_model.encode([pred])
    emb2 = embedding_model.encode([truth])
    return cosine_similarity(emb1, emb2)[0][0]


# -------------------------------
# 🔹 EVALUATION + CHART
# -------------------------------
def evaluate_and_plot(index, chunks, qa_pairs, hf_api_key):
    scores = []

    for q, truth in qa_pairs:
        retrieved = query_faiss(index, q, chunks)
        raw = get_structured_answer(retrieved, q, hf_api_key)
        parsed = safe_parse_json(raw)

        pred = parsed["answer"]

        scores.append({
            "question": q,
            "EM": exact_match(pred, truth),
            "F1": f1_score(pred, truth),
            "SIM": similarity(pred, truth)
        })

        print(f"\nQ: {q}")
        print(f"Pred: {pred}")
        print(f"Truth: {truth}")

    # ---- Plot ----
    labels = [f"Q{i+1}" for i in range(len(scores))]
    em = [s["EM"] for s in scores]
    f1 = [s["F1"] for s in scores]
    sim = [s["SIM"] for s in scores]

    x = np.arange(len(scores))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, em, width, label="Exact Match")
    plt.bar(x, f1, width, label="F1 Score")
    plt.bar(x + width, sim, width, label="Similarity")

    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("RAG Evaluation")
    plt.legend()

    plt.show()
    plt.savefig("chart.png")


# -------------------------------
# 🔹 MAIN
# -------------------------------
def main():
    pdf_path = "../data/pdfs/intelligent_systems.pdf"
    hf_api_key = os.getenv("HF_API_TOKEN")

    if not hf_api_key:
        raise ValueError("Set HF_API_TOKEN")

    print("Processing PDF...")
    data = extract_text_with_headings(pdf_path)

    chunks = chunk_by_headings(data)
    embeddings = generate_embeddings(chunks)
    index = store_in_faiss(embeddings)

    qa_pairs = [
        ("What is the course code and title?", "CS-3270: Intelligent Systems"),
        ("Who is the instructor?", "Dr. Md Shirajum Munir"),
        ("What is the instructor's email?", "mmunir@westga.edu"),
        ("When is exam 3?", "Wednesday, May 6, 12:00-2:00 PM (In person and closed book)"),
        ("When is exam 2?", "April 08, 2026, Wednesday 10:30 AM (In person and closed book)"),
        ("When is exam 1?", "February 25, 2026, Wednesday 10:30 AM (In person and closed book)"),
        ("How much does each exam weigh?", "13%"),
        ("Where is the instructor's office located?", "Technology Learning Center (TLC) 2248, 210 W Georgia Dr, Carrollton, GA 30117"),
    ]

    evaluate_and_plot(index, chunks, qa_pairs, hf_api_key)


if __name__ == "__main__":
    main()
