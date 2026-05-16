import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_store(chunks: list) -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")  # ← fix this line
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"[embedder] Created FAISS index with {len(chunks)} vectors")
    return vector_store