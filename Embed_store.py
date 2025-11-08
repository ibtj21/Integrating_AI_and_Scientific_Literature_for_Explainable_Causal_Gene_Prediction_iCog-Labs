# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

sys.stdout.reconfigure(encoding='utf-8')


class GeneEmbedder:
    """
    A modular class that embeds PubMed abstracts and stores them in a FAISS index.
    """

    def __init__(
        self,
        faiss_index_path: str = "faiss_meetta_index",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.faiss_index_path = faiss_index_path
        self.embedding_model_name = embedding_model_name
        self.embeddings = None
        self.vector_store = None
        self.data = None

    # ---------------------------
    # Step 1: Load DataFrame
    # ---------------------------
    def from_dataframe(self, df: pd.DataFrame):
        """Load abstracts directly from a pandas DataFrame."""
        if df.empty:
            raise ValueError("⚠️ DataFrame is empty. Cannot embed.")
        df["content"] = df["Title"].fillna('') + ". " + df["Abstract"].fillna('')
        self.data = df
        print(f"📚 Loaded {len(df)} abstracts directly from DataFrame.")
        return self

    # ---------------------------
    # Step 2: Initialize Model
    # ---------------------------
    def init_embeddings(self):
        print(f"🧠 Loading embedding model: {self.embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        return self.embeddings

    # ---------------------------
    # Step 3: Create FAISS Store
    # ---------------------------
    def create_vector_store(self):
        if self.data is None:
            raise ValueError("⚠️ No data loaded. Please load DataFrame first.")
        if self.embeddings is None:
            self.init_embeddings()

        print("💾 Creating FAISS index...")
        texts = self.data["content"].tolist()
        metadatas = self.data[["PMID", "Title"]].to_dict(orient="records")

        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )

        print(f"✅ FAISS index created with {len(texts)} documents.")
        return self.vector_store

    # ---------------------------
    # Step 4: Save Index
    # ---------------------------
    def save_index(self, append: bool = False):
        if self.vector_store is None:
            raise ValueError("⚠️ Vector store not created. Run create_vector_store() first.")

        if append and os.path.exists(self.faiss_index_path):
            print(f"🔁 Updating existing FAISS index at {self.faiss_index_path}...")
            existing_store = FAISS.load_local(
                self.faiss_index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            existing_store.merge_from(self.vector_store)
            existing_store.save_local(self.faiss_index_path)
        else:
            print(f"💾 Saving new FAISS index at {self.faiss_index_path}...")
            self.vector_store.save_local(self.faiss_index_path)

        print("✅ Index saved successfully.")


