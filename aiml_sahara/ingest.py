# ingest.py
"""
One-time (re-runnable) script that builds the vector store for SAHARA's RAG
first-aid / safety assistant.

Run this whenever you edit knowledge_base_docs.json:
    python ingest.py

It reads knowledge_base_docs.json, embeds each entry with a small local
sentence-transformers model (no API key needed for embeddings), and stores
the vectors in a persistent Chroma collection on disk (./chroma_db).
"""

import json
import chromadb
from chromadb.utils import embedding_functions

KB_PATH = "knowledge_base_docs.json"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "sahara_first_aid_kb"

# Free, local embedding model — good enough for a KB this size and avoids
# needing an embeddings API key just to index documents.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def build_vector_store():
    print("Loading knowledge base source documents...")
    with open(KB_PATH, "r") as f:
        docs = json.load(f)

    print(f"Loaded {len(docs)} knowledge base entries.")

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Wipe and rebuild the collection each time this script runs, so edits
    # to knowledge_base_docs.json are always reflected.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL_NAME
    )

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [doc["id"] for doc in docs]
    texts = [doc["content"] for doc in docs]
    metadatas = [
        {
            "condition": doc["condition"],
            "category": doc["category"],
            "source": doc["source"],
        }
        for doc in docs
    ]

    collection.add(ids=ids, documents=texts, metadatas=metadatas)

    print(f"Indexed {len(ids)} entries into Chroma collection '{COLLECTION_NAME}'.")
    print(f"Vector store persisted at: {CHROMA_DIR}")


if __name__ == "__main__":
    build_vector_store()
