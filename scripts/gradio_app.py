import os
import gradio as gr

from rag_pipeline import (
    extract_text_with_headings,
    chunk_by_headings,
    generate_embeddings,
    store_in_faiss,
    query_faiss,
    get_structured_answer,
    safe_parse_json
)

# -------------------------------
# Config
# -------------------------------
hf_api_key = os.getenv("HF_API_TOKEN")

# -------------------------------
# Global state
# -------------------------------
index = None
chunks = []  # stores dicts: {"text": ..., "source": ...}


# -------------------------------
# PDF Processing (multi-file)
# -------------------------------
def process_pdfs(pdf_files):
    global index, chunks

    if not pdf_files:
        return "⚠️ Please upload at least one PDF."

    all_new_chunks = []

    for pdf_file in pdf_files:
        data = extract_text_with_headings(pdf_file.name)
        doc_chunks = chunk_by_headings(data)

        # attach metadata
        doc_chunks = [
            {"text": c, "source": pdf_file.name}
            for c in doc_chunks
        ]

        all_new_chunks.extend(doc_chunks)

    if not all_new_chunks:
        return "⚠️ No text extracted."

    # embeddings use ONLY text
    texts = [c["text"] for c in all_new_chunks]
    new_embeddings = generate_embeddings(texts)

    if index is None:
        index = store_in_faiss(new_embeddings)
        chunks = all_new_chunks
    else:
        index.add(new_embeddings)
        chunks.extend(all_new_chunks)

    return f"✅ Added {len(all_new_chunks)} chunks from {len(pdf_files)} PDFs (Total: {len(chunks)})"


# -------------------------------
# Chat
# -------------------------------
def chat(query, history):
    global index, chunks

    if history is None:
        history = []

    if not query.strip():
        return history

    if index is None:
        history.append({"role": "user", "content": query})
        history.append({
            "role": "assistant",
            "content": "⚠️ Upload and process PDFs first."
        })
        return history

    # Retrieve relevant chunks
    retrieved = query_faiss(index, query, chunks)

    if not retrieved:
        answer = "⚠️ No relevant information found."
    else:
        # ✅ CRITICAL FIX: pass list[str] to pipeline
        context_chunks = [
            r["text"] if isinstance(r, dict) else r
            for r in retrieved
        ]

        raw = get_structured_answer(context_chunks, query, hf_api_key)
        parsed = safe_parse_json(raw)

        answer = parsed.get("answer", "No answer found.")

        # Optional: show sources
        sources = list(set([
            r.get("source", "unknown")
            for r in retrieved if isinstance(r, dict)
        ]))

        # if sources:
            # answer += "\n\n📄 Sources:\n" + "\n".join(sources)

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})

    return history


# -------------------------------
# Reset
# -------------------------------
def reset_all():
    global index, chunks
    index = None
    chunks = []
    return "🔄 Cleared all PDFs.", []


# -------------------------------
# UI
# -------------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 📄 Multi-PDF RAG Chat Assistant")

    with gr.Row():
        file_input = gr.File(
            label="Upload PDFs",
            file_count="multiple",
            file_types=[".pdf"]
        )
        upload_btn = gr.Button("Process PDFs")

    status = gr.Textbox(label="Status")

    upload_btn.click(
        process_pdfs,
        inputs=file_input,
        outputs=status
    )

    with gr.Row():
        reset_btn = gr.Button("Reset Knowledge Base")

    chatbot = gr.Chatbot(label="Chat")

    msg = gr.Textbox(
        placeholder="Ask a question about your PDFs...",
        label=""
    )

    msg.submit(
        chat,
        inputs=[msg, chatbot],
        outputs=chatbot
    ).then(lambda: "", None, msg)

    reset_btn.click(
        reset_all,
        outputs=[status, chatbot]
    )

# -------------------------------
# Launch
# -------------------------------
if __name__ == "__main__":
    demo.launch(inbrowser=True)

