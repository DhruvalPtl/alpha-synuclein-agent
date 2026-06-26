"""
agent/core/memory.py
────────────────────────────────────────────────────────────────────────────
Persistent Memory & Hypothesis Deduplication Engine

Uses ChromaDB to store vector embeddings of past experiments.
1. The Lead Researcher queries this to find past patterns (e.g., "what happened when we used LSTM?")
2. The Deduplicator checks if a proposed hypothesis is identical to a past one to prevent loops.
"""

import os
from pathlib import Path
import chromadb

class PersistentMemory:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.db_path = self.project_root / "data" / "chroma_db"
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Get or create the collection for experiments
        self.collection = self.client.get_or_create_collection(
            name="experiment_memory",
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )

    def add_experiment(self, exp_id: str, hypothesis: str, architecture: str, f1_macro: float, mcc: float):
        """
        Stores an experiment in the vector database.
        The document is the hypothesis and architecture text, which gets embedded.
        """
        document = f"Architecture: {architecture}. Hypothesis: {hypothesis}"
        
        metadata = {
            "exp_id": exp_id,
            "architecture": architecture,
            "f1_macro": float(f1_macro),
            "mcc": float(mcc)
        }
        
        self.collection.add(
            documents=[document],
            metadatas=[metadata],
            ids=[exp_id]
        )

    def search_experiments(self, query: str, n_results: int = 5) -> list:
        """
        Semantic search for past experiments based on a text query.
        Returns a formatted string of past insights.
        """
        # If collection is empty, return empty list
        if self.collection.count() == 0:
            return []
            
        n = min(n_results, self.collection.count())
        results = self.collection.query(
            query_texts=[query],
            n_results=n
        )
        
        return results

    def check_is_duplicate(self, new_hypothesis: str, new_architecture: str, threshold: float = 0.1) -> bool:
        """
        Hypothesis Deduplication Engine.
        Checks if the proposed idea is too similar to a past experiment.
        In Chroma, cosine distance is 0 for identical items. 
        So distance < threshold (e.g. 0.1) means it's a duplicate.
        """
        if self.collection.count() == 0:
            return False
            
        document = f"Architecture: {new_architecture}. Hypothesis: {new_hypothesis}"
        
        results = self.collection.query(
            query_texts=[document],
            n_results=1
        )
        
        distances = results.get("distances", [[1.0]])
        if distances and len(distances) > 0 and len(distances[0]) > 0:
            min_dist = distances[0][0]
            if min_dist < threshold:
                return True
                
        return False
