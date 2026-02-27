import json
from typing import List, Optional
from langchain_core.documents import Document
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVectorParams, SparseIndexParams, PointStruct, VectorParams, Distance
from tqdm import tqdm
from config.config import settings

class SimpleJSONLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(self.file_path)

        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = []

        for law in data:
            law_id = law.get("law_id", "")
            law_internal_id = law.get("id")

            for article in law.get("content", []):
                aid = article.get("aid")
                content = article.get("content_Article") or article.get("content", "")

                if not content:
                    continue

                metadata = {
                    "id": law_internal_id,
                    "law_id": law_id,
                    "aid": aid,
                }

                documents.append(
                    Document(
                        page_content=content,
                        metadata=metadata,
                    )
                )

        return documents


class QdrantVDB:

    def __init__(self,
                 collection_name: Optional[str] = None,
                 embedding_model_name: Optional[str] = None,
                 sparse_model_name: Optional[str] = None,
                 file_data_path: Optional[str] = None,
                 batch_size: Optional[int] = None):
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.batch_size = batch_size or settings.batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_name = embedding_model_name or settings.embedding_model_name
        self.embedding_model = TextEmbedding(
            model_name=model_name,
            providers=["CUDAExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
        )
        
        sparse_name = sparse_model_name or settings.sparse_embedding_model_name
        self.sparse_embedding = SparseTextEmbedding(model_name=sparse_name)
        
        self.client = None
        test_embed = list(self.embedding_model.embed(["test"]))[0]
        self.size_embedding = len(test_embed)
        self.file_data_path = file_data_path or "data/legal_corpus.json"
        self.loader = SimpleJSONLoader(file_path=self.file_data_path)
    def load_data(self) -> List[Document]:
        document = self.loader.load()
        spliter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunked_docs = spliter.split_documents(document)
        return chunked_docs
    def initialize_client(self) -> Optional[QdrantClient]:
        try:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
            return self.client
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Qdrant client: {e}") from e
    def create_collection(self):
        if self.client is None:
            print("Qdrant client is not initialized.")
            raise RuntimeError("Qdrant client initialization failed. Call initalize_client() first.")
        if self.client.collection_exists(self.collection_name):
            raise RuntimeError(f"Collection '{self.collection_name}' already exists.")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
               "dense": VectorParams(
                    size=self.size_embedding,
                    distance=Distance.COSINE
                )
            }, 
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            }
        )
        print(f"Collection '{self.collection_name}' created successfully.")
    def embedd_and_store(self, documents):
        points = []

        for i in tqdm(range(0, len(documents), self.batch_size)):
            batch = documents[i:i+self.batch_size]
            texts = [d.page_content for d in batch]

            dense = list(self.embedding_model.embed(texts))
            sparse = list(self.sparse_embedding.embed(texts))

            for j, doc in enumerate(batch):
                points.append(
                    PointStruct(
                        id=i+j,
                        vector={
                            "dense": dense[j].tolist(),
                            "sparse": {
                                "indices": sparse[j].indices.tolist(),
                                "values": sparse[j].values.tolist()
                            }
                        },
                        payload={
                            "text": doc.page_content,
                            "metadata": doc.metadata
                        }
                    )
                )

            if len(points) >= settings.upsert_batch_size:
                self.client.upsert(self.collection_name, points, wait=False)
                points = []

        if points:
            self.client.upsert(self.collection_name, points, wait=True)

    def load_data_and_store(self):
        documents = self.load_data()
        self.initialize_client()
        self.create_collection()
        self.embedd_and_store(documents)

    
