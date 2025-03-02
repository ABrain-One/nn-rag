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
        """
        Embed corpus_data and build a FAISS index.
        Optionally, save the index to disk.
        """
        self.corpus_data = corpus_data
        texts = [item["text"] for item in corpus_data]
        print(f"Embedding {len(texts)} items with model {self.model_name} ...")
        self.embeddings = self.embedder.encode(texts, batch_size=self.batch_size, convert_to_numpy=True)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        print("FAISS index built. Total items:", self.index.ntotal)

        if self.index_path:
            print(f"Saving FAISS index to {self.index_path} ...")
            faiss.write_index(self.index, self.index_path)

    def load_index(self, index_path, corpus_data):
        """
        Load a FAISS index from disk.
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"{index_path} not found. Build the index first.")
        print(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)
        self.corpus_data = corpus_data
        self.index_path = index_path

    def search(self, query, top_k=5):
        """
        Embed the query and search the FAISS index.
        Returns the top-k results with metadata and distances.
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
