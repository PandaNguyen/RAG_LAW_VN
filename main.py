from src.retrieve.retrieve_rerank import Retriever
from src.vectorstores.qdrant_store import QdrantRetriever
from src.generation.rag import RAGPipeline
from src.indexing.qdrantvdb import QdrantVDB


def main():
    """
    Example usage of RAG pipeline for Vietnamese legal Q&A
    
    Uncomment the section you want to test:
    1. Index documents to Qdrant
    2. Query RAG pipeline
    """
    
    # ===== OPTION 1: Index documents to Qdrant =====
    # loader = QdrantVDB(
    #     collection_name="legal_documents",
    #     file_path="data/legal_corpus.json",
    # )
    # docs = loader.load_data()
    # print(f"Loaded {len(docs)} documents")
    # loader.load_data_and_store()
    
    # ===== OPTION 2: Query RAG pipeline =====
    # Setup retriever
    vector_store = QdrantRetriever(
        collection_name="legal_documents",
        retrieve_mode="hybrid"
    )
    retriever = Retriever(
        vector_store=vector_store,
        search_mode="hybrid",
        top_k=5
    )
    
    # Setup RAG pipeline with Gemini
    rag = RAGPipeline(retriever=retriever)
    
    # Example query
    question = "Thưa luật sư tôi có đăng ký kết hôn trên pháp luật nhưng nay vợ chồng bỏ nhau theo phong tục tập quán như vậy tôi có được phép kết hôn với người khác không ạ?"
    print(f"\nCÂU HỎI: {question}\n")
    
    # Get formatted response with sources
    response = rag.query_with_sources(question)
    print(response)


if __name__ == "__main__":
    main()
