from PyPDF2 import PdfReader
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_with_headings(pdf_path):
    reader = PdfReader(pdf_path)
    data = []
    current_section = {"heading": "Document", "content": ""}

    for page in reader.pages:
        text = page.extract_text()
        if not text:
            continue

        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect uppercase headings (less strict)
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
        content = section["content"].split()
        for i in range(0, len(content), max_chunk_size):
            chunk = " ".join(content[i:i + max_chunk_size])
            chunks.append(f"{heading}\n{chunk}")
    return chunks

def generate_embeddings(chunks):
    return embedding_model.encode(chunks, normalize_embeddings=True)
    

def store_in_faiss(embeddings):
    embeddings = np.array(embeddings)
    if len(embeddings.shape) != 2:
        raise ValueError("Embeddings are empty or invalid. Check chunk generation.")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings))
    return index

def query_faiss(index, query, chunks):
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(np.array(query_embedding), 5)  # Find top 5 matches
    return [chunks[i] for i in indices[0]]


def get_answer_with_huggingface(context, query, hf_api_key):
    API_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {"Authorization": f"Bearer {hf_api_key}",
               "Content-Type": "application/json"}

    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct:novita",
        "messages": [
            {
                "role": "system",
                "content": "Answer ONLY using the provided context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ],
       
        "max_new_tokens": 300,
        "temperature": 1.0,
        "return_full_text": False
        
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    # Basic error handling
    if response.status_code != 200:
        return f"Error: {response.json()}"

    result = response.json()

    return result["choices"][0]["message"]["content"].strip()

def main():
    pdf_path = "../data/pdfs/intelligent_systems.pdf"
    hf_api_key = "hf_api_key"
    
    data = extract_text_with_headings(pdf_path)
    chunks = chunk_by_headings(data)
    embeddings = generate_embeddings(chunks)
    index = store_in_faiss(embeddings)
    
    while True:
        user_query = input("Enter your question: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        relevant_chunks = query_faiss(index, user_query, chunks)
        context = "\n".join(relevant_chunks)
        
        answer = get_answer_with_huggingface(context, user_query, hf_api_key)
        print(f"\n Answer: {answer}")

if __name__ == "__main__":
    main()
