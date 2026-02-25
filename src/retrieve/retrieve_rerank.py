
from src.vectorstores.qdrant_store import QdrantVectorStore

class Retriever:

    def __init__(self,
                 vector_store: QdrantVectorStore,
                 search_mode = "hybrid",
                 top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k
        self.search_mode = search_mode
    def search(self, query: str, top_k : int = 5):
        search_results = self.vector_store.get_relevant_documents(query, k=top_k)
        # print(search_results)
        return search_results


Retriever = Retriever
