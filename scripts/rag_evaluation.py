import numpy as np
import requests
import re
import json
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# -------------------------------
# 🔹 Similarity
# -------------------------------
def semantic_similarity(a, b):
    emb1 = embedding_model.encode([a], normalize_embeddings=True)
    emb2 = embedding_model.encode([b], normalize_embeddings=True)
    return np.dot(emb1, emb2.T)[0][0]


# -------------------------------
# 🔹 LLM Judge
# -------------------------------
def llm_judge(prompt, hf_api_key):
    API_URL = "https://router.huggingface.co/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct:novita",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    try:
        return response.json()["choices"][0]["message"]["content"]
    except:
        return "0"


def extract_number(text):
    match = re.search(r"\d+(\.\d+)?", text)
    return float(match.group()) if match else 0.0


# -------------------------------
# 🔹 Metrics
# -------------------------------
def answer_correctness(q, pred, gt, key):
    prompt = f"""
    Question: {q}
    Ground Truth: {gt}
    Answer: {pred}

    Score correctness from 0 to 1.
    """
    return extract_number(llm_judge(prompt, key))


def context_precision(context_chunks, citations):
    if not context_chunks:
        return 0
    return len(set(citations)) / len(context_chunks)


def verify_claims(claims, context, key):
    scores = []

    for claim in claims:
        prompt = f"""
        Context:
        {context}

        Claim:
        {claim}

        Supported? 1 or 0 only.
        """
        scores.append(extract_number(llm_judge(prompt, key)))

    return np.mean(scores) if scores else 0


def citation_accuracy(citations, context_chunks):
    if not citations:
        return 0
    valid = sum(1 for c in citations if c < len(context_chunks))
    return valid / len(citations)


# -------------------------------
# 🔹 JSON Parsing
# -------------------------------
def safe_parse_json(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"answer": "", "citations": [], "claims": []}


# -------------------------------
# 🔹 MAIN EVALUATION
# -------------------------------
def evaluate_rag_advanced(index, chunks, qa_pairs, hf_api_key, query_fn):
    results = []

    for i, (question, ground_truth) in enumerate(qa_pairs, 1):
        print(f"\n🔍 Q{i}: {question}")
        
        retrieved_chunks = query_fn(index, question, chunks)
        from pdf_plum import get_structured_answer

        raw_output = get_structured_answer(
            retrieved_chunks,
            question,
            hf_api_key
        )

        parsed = safe_parse_json(raw_output)

        answer = parsed.get("answer", "")
        citations = parsed.get("citations", [])
        claims = parsed.get("claims", [])
        print("Answer:", answer)

        context = "\n".join(retrieved_chunks)
        correctness = answer_correctness(question, answer, ground_truth, hf_api_key)
        precision = context_precision(retrieved_chunks, citations)
        hallucination = verify_claims(claims, context, hf_api_key)
        citation_score = citation_accuracy(citations, retrieved_chunks)
        similarity = semantic_similarity(answer, ground_truth)
        
        results.append({
            "question": question,
            "answer": answer,
            "correctness": float(correctness),
            "context_precision": float(precision),
            "hallucination_score": float(hallucination),
            "citation_accuracy": float(citation_score),
            "semantic_similarity": float(similarity)
        })
    return results


# -------------------------------
# 🔹 Summary
# -------------------------------
def summarize_advanced(results):
    keys = [
        "correctness",
        "context_precision",
        "hallucination_score",
        "citation_accuracy",
        "semantic_similarity"
    ]

    print("\n📊 FINAL RESULTS")
    for k in keys:
        avg = np.mean([r[k] for r in results])
        print(f"{k}: {avg:.3f}")
