# src/cyborg_client.py

import numpy as np


class CyborgClient:
    def __init__(self):
        print("[CyborgClient] Using clean in-memory index.")
        self.indexes = {}

    # ------------------------------------------------------
    # Create an index
    # ------------------------------------------------------
    def create_encrypted_index(self, name: str):
        self.indexes[name] = {
            "vectors": [],
            "dim": None  # Will be set on first upsert
        }
        return name

    def delete_index(self, name: str):
        if name in self.indexes:
            del self.indexes[name]

    # ------------------------------------------------------
    # Insert vectors, set dim on first insert
    # ------------------------------------------------------
    def upsert_vectors(self, index_name, vectors):
        idx = self.indexes[index_name]

        if not vectors:
            return  # Nothing to insert

        # Set dimension if first insert
        if idx["dim"] is None:
            idx["dim"] = len(vectors[0]["vector"])
            print(f"[CyborgClient] Index dim set to {idx['dim']}")

        # Validate all vectors
        for v in vectors:
            if len(v["vector"]) != idx["dim"]:
                raise ValueError(
                    f"Vector dim mismatch: expected {idx['dim']} got {len(v['vector'])}"
                )
            idx["vectors"].append(v)

    # ------------------------------------------------------
    # Query vectors using cosine similarity
    # ------------------------------------------------------
    def query_index(self, index_name, query_vector, k=5):
        idx = self.indexes[index_name]

        # If index is empty, just return empty results
        if not idx["vectors"]:
            return {"results": []}

        if len(query_vector) != idx["dim"]:
            raise ValueError(
                f"Query vector dim mismatch: expected {idx['dim']} got {len(query_vector)}"
            )

        # Convert to numpy for similarity
        q = np.array(query_vector)

        scored = []
        for item in idx["vectors"]:
            v = np.array(item["vector"])
            sim = np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-10)
            scored.append((sim, item))

        # Sort by similarity desc
        scored.sort(key=lambda x: x[0], reverse=True)

        return {
            "results": [s[1] for s in scored[:k]]
        }








