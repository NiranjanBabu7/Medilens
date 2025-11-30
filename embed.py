# src/embed.py

from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict

# ------------------------------------------------------
# 🚀 SINGLETON EMBEDDER (Loads only once)
# ------------------------------------------------------
class Embedder:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        self.model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
        print(f"[Embedder] Loading model: {self.model_name}")

        # Force CPU (safe for all systems)
        device = torch.device("cpu")

        # Load model
        self.model = SentenceTransformer(self.model_name, device=device)

        # MiniLM-L3-v2 dimension is ALWAYS 384
        self.dim = 384

    # ------------------------------------------------------
    # 🔹 Generate embeddings in consistent 384-dim form
    # ------------------------------------------------------
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            batch_size=4,
            show_progress_bar=False
        )

        # Convert numpy → python list
        return [v.tolist() for v in vectors]


# ------------------------------------------------------
# 🔹 Build embedding objects for Cyborg DB
# ------------------------------------------------------
def build_embeddings(records: List[Dict], embedder: Embedder) -> List[Dict]:
    output = []

    for r in records:
        text = r.get("text_masked", "")

        vec = embedder.embed_texts([text])[0]  # ALWAYS 384-dim

        output.append({
            "id": r["anon_id"],
            "vector": vec,
            "content": text,
            "metadata": {
                "timestamp": r.get("timestamp", "")
            }
        })

    return output



