# -*- coding: utf-8 -*-
import os
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import sys
sys.stdout.reconfigure(encoding='utf-8')
# -------------------------
# Keywords to compare
# -------------------------
KEYWORDS = [
    "Alzheimer", "ABCA7", "BIN1", "CD2AP", "CD33",
    "CLU", "CR1", "EPHA1", "MS4A", "PICALM",
    "SORL1", "TREM2"
]

class SimilaritySearcher:
    """
    Semantic-only search on a FAISS vector store with keyword match reporting.
    """

    def __init__(
        self,
        faiss_index_path: str = "faiss_meetta_index",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.faiss_index_path = faiss_index_path
        self.embedding_model_name = embedding_model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.vector_store = None

    def load_index(self):
        """Load existing FAISS index from disk."""
        if not os.path.exists(self.faiss_index_path):
            raise FileNotFoundError(f"❌ FAISS index not found at {self.faiss_index_path}")
        self.vector_store = FAISS.load_local(
            self.faiss_index_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        print(f"✅ FAISS index loaded from {self.faiss_index_path}")

    @staticmethod
    def get_matched_keywords(text: str) -> List[str]:
        """
        Returns a list of keywords found in the given text.
        """
        text_lower = text.lower() if text else ""
        return [kw for kw in KEYWORDS if kw.lower() in text_lower]

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for top_k most similar documents (semantic only).
        Adds reporting of keyword matches (for comparison purposes only).
        """
        if self.vector_store is None:
            self.load_index()

        results = self.vector_store.similarity_search(query, k=top_k)
        top_docs = []

        for doc in results:
            content = doc.page_content
            metadata = doc.metadata
            pmid = metadata.get("PMID", "N/A")  # Add PMID
            # Count keyword matches in the content
            matched_keywords = self.get_matched_keywords(content)
            keyword_count = len(matched_keywords)

            top_docs.append({
                "content": content,
                "metadata": metadata,
                "pmid": pmid,
                "keyword_count": keyword_count,
                "matched_keywords": matched_keywords
            })

        # -------------------------
        # Print results for comparison
        # -------------------------
        print(f"🔍 Found top {len(top_docs)} documents for query: '{query}'\n")
        for i, doc in enumerate(top_docs, 1):
            print(f"{i}. PMID: {doc['pmid']}")
            print(f"   Content preview: {doc['content'][:100]}...")
            print(f"   Keyword matches ({doc['keyword_count']}): {doc['matched_keywords']}\n")

        return top_docs


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    searcher = SimilaritySearcher()
    search_query = (
        "Alzheimer phenotype AND "
        "(ABCA7 OR BIN1 OR CD2AP OR CD33 OR CLU OR CR1 OR EPHA1 "
        "OR MS4A OR PICALM OR SORL1 OR TREM2)"
    )
    results = searcher.search(search_query, top_k=5)
