# -*- coding: utf-8 -*-
import os
import sys
from PubMedRetriever import PubMedRetriever
from Embed_store import GeneEmbedder
from SimilaritySearch import SimilaritySearcher
from Prediction import BioGPTGeneRanker

sys.stdout.reconfigure(encoding='utf-8')

def main():
    # Step 1: Retrieve PubMed abstracts
    search_query = "Alzheimer phenotype AND (ABCA7 OR BIN1 OR CD2AP OR CD33 OR CLU OR CR1 OR EPHA1 OR MS4A OR PICALM OR SORL1 OR TREM2)"
    retriever = PubMedRetriever(os.getenv("API_KEY"))  # uses default API key if none in environment
    df = retriever.retrieve(search_query, max_results=20)

    if df.empty:
        print("⚠️ No data retrieved. Exiting.")
        return

    # Step 2: Embed and store
    embedder = GeneEmbedder(
        faiss_index_path="faiss_meetta_index",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    embedder.from_dataframe(df)
    embedder.init_embeddings()
    embedder.create_vector_store()
    embedder.save_index(append=False)

    # Step 3: Similarity Search
    searcher = SimilaritySearcher(faiss_index_path="faiss_meetta_index")
    top_docs = searcher.search("Predict the causal gene involved in Alzheimer for these loci", top_k=5)

    # Step 4: Predict causal gene using BioGPT
    candidate_genes = ["ABCA7", "BIN1", "CD2AP", "CD33", "CLU", "CR1", "EPHA1", "MS4A", "PICALM", "SORL1", "TREM2"]
    predictor = BioGPTGeneRanker(model_name="microsoft/biogpt", mc_samples=5)
    prediction = predictor.predict(top_docs, "Alzheimer", candidate_genes)

    print("🎉 Final Causal Gene Prediction:")
    print("Predicted gene:", prediction["predicted_gene"])
    print("Confidence score:", prediction["confidence_score"])
    print("\nExplanation:\n", prediction["explanation"])

if __name__ == "__main__":
    main()
