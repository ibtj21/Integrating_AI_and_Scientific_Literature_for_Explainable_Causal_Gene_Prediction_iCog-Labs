# -*- coding: utf-8 -*-
import os
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class SimilaritySearcher:
    """
    A class to perform semantic similarity search on a FAISS vector store.
    Returns top N most relevant documents for a given query.
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
        self.vector_store = FAISS.load_local(self.faiss_index_path, self.embeddings, allow_dangerous_deserialization=True)
        print(f"✅ FAISS index loaded from {self.faiss_index_path}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for the top_k most similar documents to the query.
        Returns a list of dicts containing 'content' and metadata.
        """
        if self.vector_store is None:
            self.load_index()

        results = self.vector_store.similarity_search(query, k=top_k)
        top_docs = []
        for doc in results:
            top_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        print(f"🔍 Found top {len(top_docs)} documents for query: '{query}'")
        return top_docs



if __name__ == "__main__":
    # ---------------------------
    # Step 0: Import necessary classes
    # ---------------------------
    from Embed_store import GeneEmbedder  
    from SimilaritySearch import SimilaritySearcher

    # ---------------------------
    # Step 1: Load FAISS index from GeneEmbedder
    # ---------------------------
    faiss_index_path = "alzheimer_pubmed_faiss_index"  # use the path where GeneEmbedder saved the index

    # Initialize the searcher
    searcher = SimilaritySearcher(faiss_index_path=faiss_index_path)

    # Explicitly load the FAISS index
    searcher.load_index()

    # ---------------------------
    # Step 2: Run a test query
    # ---------------------------
    test_gene = "ABCA7"
    test_phenotype = "Alzheimer phenotype"
    query = f"{test_phenotype} AND {test_gene}"
    top_k = 3

    results = searcher.search(query=query, top_k=top_k)

    # ---------------------------
    # Step 3: Print results
    # ---------------------------
    print(f"\n🔍 Top {top_k} abstracts for query: '{query}'\n")
    for i, doc in enumerate(results, 1):
        print(f"{i}. PMID: {doc['metadata'].get('PMID', 'N/A')}")
        print(f"   Title: {doc['metadata'].get('Title', 'No title')}")
        print(f"   Abstract preview: {doc['content'][:300]}...\n")
