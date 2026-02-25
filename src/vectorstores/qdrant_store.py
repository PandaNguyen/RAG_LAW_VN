"""
Base retriever interface for document retrieval.
"""
from abc import ABC, abstractmethod
from typing import List, Any
from langchain_core.documents import Document


class BaseRetriever(ABC):
    """
    Abstract base class for document retrievers.
    Provides a standard interface for all retrieval implementations.
    """

    def __init__(self, name: str = "base_retriever"):
        """
        Initialize base retriever.
        
        Args:
            name: Name of the retriever
        """
        self.name = name

    @abstractmethod
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        pass

    @abstractmethod
    def as_retriever(self, **kwargs) -> Any:
        """
        Return a LangChain-compatible retriever.
        
        Args:
            **kwargs: Additional arguments for retriever configuration
            
        Returns:
            LangChain retriever instance
        """
        pass

    def info(self) -> str:
        """
        Get retriever information.
        
        Returns:
            String describing the retriever
        """
        return f"Retriever: {self.name}"

from typing import List, Literal, Any, Optional
from dotenv import load_dotenv
from src.embedding.embedd_data import VietnameseLegalEmbedding
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
import os
from qdrant_client import QdrantClient
from config.config import settings
load_dotenv()



class QdrantRetriever(BaseRetriever):
    """
    Qdrant-based document retriever.
    """

    def __init__(self,
                top_k: Optional[int] = None,
                collection_name: Optional[str] = None,
                size_embedding: Optional[int] = None,
                vector_embedding: Any = None,
                sparse_vector_embedding: Any = None,
                retrieve_mode: Optional[Literal['dense', 'sparse', 'hybrid']] = None):
        super().__init__(name="qdrant_retriever")
 
        self.top_k = top_k or settings.default_top_k
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.size_embedding = size_embedding or settings.embedding_size
        
        self.vector_embedding = vector_embedding or VietnameseLegalEmbedding(
            model_name=settings.embedding_model_name
        )
        self.sparse_vector_embedding = sparse_vector_embedding or FastEmbedSparse(
            model_name=settings.sparse_embedding_model_name
        )
        self.retrieve_mode = retrieve_mode or "hybrid"
        
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self.store = QdrantVectorStore.from_existing_collection(
            collection_name=self.collection_name,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            embedding=self.vector_embedding,
            vector_name='dense',
            sparse_embedding=self.sparse_vector_embedding,
            sparse_vector_name='sparse',
            content_payload_key='text',
            metadata_payload_key='metadata',
            retrieval_mode=self.retrieve_mode
        )
    def get_relevant_documents(self, query: str, k: Optional[int] = None) -> List[Document]:
        top_k = k if k is not None else self.top_k
        return self.store.similarity_search(query, k=top_k)

    def as_retriever(self, **kwargs) -> Any:
        """
        Return a LangChain-compatible retriever.
        
        Args:
            **kwargs: Additional arguments for retriever configuration
            
        Returns:
            LangChain retriever instance
        """
        return self.store.as_retriever(**kwargs)
