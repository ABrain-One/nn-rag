# retrieval.py

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class CodeRetrieval:
    def __init__(self, model_name, batch_size=8, index_path=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.index_path = index_path
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.corpus_data = None
        self.embeddings = None

    def build_index(self, corpus_data):
        self.corpus_data = corpus_data
        texts = [item["text"] for item in corpus_data]
        print(f"Embedding {len(texts)} items with model {self.model_name} ...")
        self.embeddings = self.embedder.encode(texts, batch_size=self.batch_size, convert_to_numpy=True)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        print("FAISS index built. Total items:", self.index.ntotal)

        if self.index_path:
            # Create the directory if needed
            index_dir = os.path.dirname(self.index_path)
            os.makedirs(index_dir, exist_ok=True)
            
            print(f"Saving FAISS index to {self.index_path} ...")
            faiss.write_index(self.index, self.index_path)

    def load_index(self, index_path, corpus_data):
        """
        Loads an existing FAISS index from disk.
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"{index_path} not found. Build the index first.")
        print(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)
        self.corpus_data = corpus_data
        self.index_path = index_path

    def search(self, query, top_k=5):
        """
        Searches the FAISS index with a query and returns the top_k results.
        """
        if self.index is None:
            raise ValueError("Index not built or loaded. Please call build_index() or load_index() first.")
        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_emb, top_k)
        results = []
        for rank, idx in enumerate(indices[0]):
            snippet = self.corpus_data[idx]
            results.append({
                "text": snippet["text"],
                "metadata": snippet["metadata"],
                "distance": float(distances[0][rank])
            })
        return results
