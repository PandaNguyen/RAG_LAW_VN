from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "legal_documents"
    
    # Embeddings
    embedding_model_name: str = "AITeamVN/Vietnamese_Embedding_v2"
    sparse_embedding_model_name: str = "Qdrant/bm25"
    embedding_size: int = 1024
    
    # Chunking
    chunk_size: int = 2048
    chunk_overlap: int = 128
    
    # Retrieval
    default_top_k: int = 5
    batch_size: int = 16
    upsert_batch_size: int = 500
    
    # Generation (for Phase 3) - Using Gemini
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.0
    google_api_key: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()
