# app/robust_embedding.py
import requests
from typing import List
from langchain_core.embeddings import Embeddings

class RobustLMStudioEmbeddings(Embeddings):
    """
    A custom embedding class that uses 'requests' (which we know works) 
    instead of 'httpx' (which is failing).
    """
    def __init__(self, base_url: str, model: str):
        # Ensure URL doesn't end with slash
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts (used for adding data)."""
        url = f"{self.base_url}/embeddings"
        
        # Construct payload exactly like your working test script
        payload = {
            "model": self.model,
            "input": texts
        }

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            # Extract embeddings ensuring order is preserved
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            print(f"❌ Embedding Error: {e}")
            raise e

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query (used for asking questions)."""
        return self.embed_documents([text])[0]