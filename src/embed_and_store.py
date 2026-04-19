import json
import os
from pathlib import Path

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
except ImportError:
    print("Please install requirements: pip install langchain-chroma langchain-huggingface sentence-transformers chromadb")
    exit(1)

CHUNKS_PATH = Path("data/parsed/2025_GFR_chunks.json")
DB_DIR = Path("/tmp/gfr_chroma_db")          # write to local /tmp (NAS breaks SQLite)
NAS_DB_DIR = Path("data/chroma_db")           # copy here after embedding for persistence

def main():
    if not CHUNKS_PATH.exists():
        print(f"File not found: {CHUNKS_PATH}")
        return
        
    print(f"Loading chunks from {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    documents = []
    for chunk in chunks:
        # Construct a rich page content block to ensure the embedding captures context
        page_content = f"Chapter: {chunk.get('chapter', '')}\nRule: {chunk.get('rule_number', '')}\nTitle: {chunk.get('title', '')}\n\nContent:\n{chunk.get('content', '')}"
        
        metadata = {
            "chunk_id": chunk["id"],
            "rule_number": str(chunk.get("rule_number", "")),
            "chapter": chunk.get("chapter", ""),
            "title": chunk.get("title", ""),
        }
        
        # Merge extra metadata and filter out None values
        if "metadata" in chunk:
            for k, v in chunk["metadata"].items():
                if v is not None:
                    metadata[k] = v if not isinstance(v, bool) else str(v)
                    
        documents.append(Document(page_content=page_content, metadata=metadata))
        
    print(f"Prepared {len(documents)} documents for vectorization.")
    
    print("Initializing HuggingFace Embeddings (bge-large-en-v1.5)...")
    # Setting model_kwargs to use cpu due to VRAM constraints
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    
    print(f"Building Chroma vector store at {DB_DIR}...")
    # Generate embeddings and store them
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
        collection_name="gfr_2025"
    )
    
    print("Vector database created successfully!")
    
    # Copy to NAS for persistence across reboots
    import shutil
    if NAS_DB_DIR.exists():
        shutil.rmtree(NAS_DB_DIR)
    shutil.copytree(DB_DIR, NAS_DB_DIR)
    print(f"Copied vector DB to NAS at {NAS_DB_DIR}")
    
    # Run a quick test query
    query = "What are the rules for procurement of goods?"
    print(f"\n--- Testing Retrieval for Query: '{query}' ---")
    results = vector_store.similarity_search(query, k=2)
    
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}: Rule {doc.metadata.get('rule_number')} - {doc.metadata.get('title')}")
        print(f"Snippet: {doc.page_content[:150].replace(chr(10), ' ')}...")

if __name__ == '__main__':
    main()
