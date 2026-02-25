from typing import Any, Dict, List, Optional

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from config.config import settings


class RAGPipeline:
    """Complete RAG pipeline for Vietnamese legal Q&A using Gemini"""
    
    def __init__(
        self,
        retriever,
        llm: Optional[ChatGoogleGenerativeAI] = None,
    ):
        """
        Initialize RAG pipeline
        
        Args:
            retriever: Document retriever instance
            llm: Optional Gemini LLM instance (will create default if None)
        """
        self.retriever = retriever
        self.llm = llm or self._create_default_llm()
        self.chain = self._build_chain()

    def _extract_documents(self, question: str) -> List[Document]:
        """Retrieve and normalize documents from the configured retriever."""
        top_k = getattr(self.retriever, "top_k", settings.default_top_k)

        if hasattr(self.retriever, "get_relevant_documents"):
            raw_docs = self.retriever.get_relevant_documents(question, k=top_k)
        elif hasattr(self.retriever, "vector_store") and hasattr(
            self.retriever.vector_store, "get_relevant_documents"
        ):
            raw_docs = self.retriever.vector_store.get_relevant_documents(
                question, k=top_k
            )
        elif hasattr(self.retriever, "search"):
            raw_docs = self.retriever.search(question, top_k=top_k) or []
        else:
            raise ValueError(
                "Retriever must implement get_relevant_documents(...) or search(...)."
            )

        docs: List[Document] = []
        for item in raw_docs or []:
            if isinstance(item, Document):
                docs.append(item)
            elif isinstance(item, tuple) and item and isinstance(item[0], Document):
                docs.append(item[0])
        return docs

    def _format_context(self, documents: List[Document]) -> str:
        """Build LLM context text from retrieved documents."""
        if not documents:
            return "Không có văn bản pháp luật liên quan trong dữ liệu truy xuất."

        chunks: List[str] = []
        for i, doc in enumerate(documents, 1):
            law_id = doc.metadata.get("law_id", "N/A")
            aaid = doc.metadata.get("aaid", "N/A")
            content = doc.page_content.strip()
            chunks.append(f"[{i}] {law_id} - Điều {aaid}\n{content}")
        return "\n\n".join(chunks)
    
    def _create_default_llm(self) -> ChatGoogleGenerativeAI:
        """Create default Gemini LLM from settings"""
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            google_api_key=settings.google_api_key,
        )
    
    def _build_chain(self):
        prompt_template = """Bạn là trợ lý AI chuyên về luật pháp Việt Nam với nhiệm vụ tư vấn pháp luật chính xác.

NGUYÊN TẮC TRẢ LỜI:
1. Chỉ dựa vào các văn bản pháp luật được cung cấp trong phần CONTEXT
2. Trích dẫn cụ thể điều luật, khoản, điểm (nếu có)
3. Nếu không có thông tin trong context, hãy nói rõ "Tôi không tìm thấy thông tin về vấn đề này trong các văn bản pháp luật được cung cấp"
4. Trả lời bằng tiếng Việt, rõ ràng, dễ hiểu
5. Tránh đưa ra ý kiến cá nhân hoặc suy đoán

CONTEXT (Các văn bản pháp luật liên quan):
{context}

CÂU HỎI: {question}

TRẢ LỜI (theo cấu trúc sau):
- Căn cứ pháp lý: [Trích dẫn điều luật cụ thể]
- Nội dung quy định: [Giải thích nội dung]
- Lưu ý: [Các điểm cần chú ý nếu có]

Trả lời:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        answer_chain = (
            {
                "context": itemgetter("context"),
                "question": itemgetter("question"),
            }
            | PROMPT
            | self.llm
            | StrOutputParser()
        )

        chain = (
            RunnablePassthrough.assign(question=itemgetter("query"))
            .assign(source_documents=itemgetter("query") | RunnableLambda(self._extract_documents))
            .assign(context=itemgetter("source_documents") | RunnableLambda(self._format_context))
            .assign(result=answer_chain)
            | RunnableLambda(
                lambda x: {
                    "result": x["result"],
                    "source_documents": x["source_documents"],
                }
            )
        )
        return chain

    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG pipeline
        
        Args:
            question: User question in Vietnamese
            
        Returns:
            Dict with:
                - 'result': Answer from LLM
                - 'source_documents': List of source documents used
        """
        return self.chain.invoke({"query": question})
    
    def query_with_sources(self, question: str) -> str:
        """
        Query and format response with source citations
        
        Args:
            question: User question
            
        Returns:
            Formatted string with answer and sources
        """
        response = self.query(question)
        
        # Format answer
        output = f"CÂU TRẢ LỜI:\n{response['result']}\n\n"
        
        # Add sources
        if response['source_documents']:
            output += "NGUỒN THAM KHẢO:\n"
            for i, doc in enumerate(response['source_documents'], 1):
                law_id = doc.metadata.get('law_id', 'N/A')
                aid = doc.metadata.get('aid', 'N/A')
                output += f"{i}. {law_id} - Điều {aid}\n"
        
        return output
    
