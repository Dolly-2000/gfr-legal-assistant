"""
Hybrid Retriever: BM25 (keyword) + ChromaDB (semantic)
========================================================
Combines sparse keyword matching with dense semantic search
using Reciprocal Rank Fusion (RRF) for improved retrieval.

This is a key thesis contribution: demonstrating that hybrid
retrieval outperforms pure semantic search for legal documents
where exact rule numbers and legal terms matter.
"""

import json
import os
import shutil
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class HybridRetriever:
    """Combines BM25 keyword search with ChromaDB semantic search."""

    def __init__(self, vectorstore, chunks_path="data/parsed/2025_GFR_chunks.json", k=5, bm25_weight=0.4, semantic_weight=0.6):
        self.vectorstore = vectorstore
        self.k = k
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

        # Load chunks for BM25
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        self.documents = []
        self.bm25_corpus = []
        for chunk in chunks:
            page_content = f"Chapter: {chunk.get('chapter', '')}\nRule: {chunk.get('rule_number', '')}\nTitle: {chunk.get('title', '')}\n\nContent:\n{chunk.get('content', '')}"
            metadata = {
                "chunk_id": chunk["id"],
                "rule_number": str(chunk.get("rule_number", "")),
                "chapter": chunk.get("chapter", ""),
                "title": chunk.get("title", ""),
            }
            self.documents.append(Document(page_content=page_content, metadata=metadata))
            # Tokenize for BM25
            self.bm25_corpus.append(page_content.lower().split())

        self.bm25 = BM25Okapi(self.bm25_corpus)

    def invoke(self, query: str):
        """Retrieve documents using hybrid BM25 + semantic search with RRF."""
        # 1. BM25 keyword search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

        # 2. Semantic search via ChromaDB
        semantic_results = self.vectorstore.similarity_search_with_relevance_scores(query, k=self.k * 3)

        # Build doc_id -> rank maps
        bm25_rank_map = {}
        for rank, idx in enumerate(bm25_ranked[:self.k * 3]):
            doc_id = self.documents[idx].metadata["chunk_id"]
            bm25_rank_map[doc_id] = rank + 1  # 1-indexed

        semantic_rank_map = {}
        semantic_doc_map = {}
        for rank, (doc, score) in enumerate(semantic_results):
            doc_id = doc.metadata.get("chunk_id", f"sem_{rank}")
            semantic_rank_map[doc_id] = rank + 1
            semantic_doc_map[doc_id] = doc

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_constant = 60  # standard RRF constant
        all_doc_ids = set(bm25_rank_map.keys()) | set(semantic_rank_map.keys())

        rrf_scores = {}
        for doc_id in all_doc_ids:
            bm25_rrf = self.bm25_weight / (rrf_constant + bm25_rank_map.get(doc_id, 1000))
            sem_rrf = self.semantic_weight / (rrf_constant + semantic_rank_map.get(doc_id, 1000))
            rrf_scores[doc_id] = bm25_rrf + sem_rrf

        # 4. Sort by RRF score and return top-k
        ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:self.k]

        # Build final document list
        # Prefer semantic_doc_map (has embeddings-based content), fall back to BM25 docs
        doc_lookup = {d.metadata["chunk_id"]: d for d in self.documents}
        doc_lookup.update({did: doc for did, doc in semantic_doc_map.items()})

        results = []
        for doc_id in ranked_ids:
            if doc_id in doc_lookup:
                results.append(doc_lookup[doc_id])

        return results


def build_hybrid_retriever(k=5, bm25_weight=0.4, semantic_weight=0.6):
    """Factory function to create a HybridRetriever with standard config."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"}
    )

    local_db_dir = "/tmp/gfr_chroma_db"
    nas_db_dir = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
    if not os.path.exists(local_db_dir):
        shutil.copytree(nas_db_dir, local_db_dir)

    vectorstore = Chroma(
        persist_directory=local_db_dir,
        embedding_function=embeddings,
        collection_name="gfr_2025"
    )

    chunks_path = os.path.join(os.path.dirname(__file__), "..", "data", "parsed", "2025_GFR_chunks.json")

    return HybridRetriever(
        vectorstore=vectorstore,
        chunks_path=chunks_path,
        k=k,
        bm25_weight=bm25_weight,
        semantic_weight=semantic_weight
    ), vectorstore, embeddings
