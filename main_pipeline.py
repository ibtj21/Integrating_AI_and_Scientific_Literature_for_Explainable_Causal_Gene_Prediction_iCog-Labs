from PubMedRetriever import PubMedRetriever
from Embed_store import GeneEmbedder
from SimilaritySearch import SimilaritySearcher
from Prediction import CausalGenePredictor

def main():
    search_query = "Obesity AND (MC4R OR BDNF OR PCSK1 OR POMC OR SH2B1 OR LEPR OR NTRK2)"

    # Step 1: Retrieve PubMed abstracts
    retriever = PubMedRetriever(api_key="6a41a7035aeb228d716b5db84d688726d908")
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
    top_docs = searcher.search("Predict the causal gene involved in Obesity for these loci", top_k=5)

    # Step 4: Predict causal gene using BioGPT
    predictor = CausalGenePredictor(model_name="microsoft/biogpt")
    candidate_genes = ["MC4R", "BDNF", "PCSK1", "POMC", "SH2B1", "LEPR", "NTRK2"]
    prediction = predictor.predict("Obesity", candidate_genes, top_docs)

    print("🎉 Final Causal Gene Prediction:", prediction)

if __name__ == "__main__":
    main()
