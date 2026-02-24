from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class VietnameseLegalEmbedding(Embeddings):
    """
    Embedding model for Vietnamese legal documents.
    Uses Vietnamese word segmentation and sentence transformers.
    """
    def __init__(self, model_name: str = "AITeamVN/Vietnamese_Embedding_v2"):
        """
        Initialize Vietnamese legal embedding model.
        
        Args:
            model_name: HuggingFace model name for embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 2048
        print(f"load model embedding: {model_name}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embedding vectors (normalized)
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector (normalized)
        """
        embedding = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return embedding[0].tolist()    