<div align="center">

# Vietnamese Legal RAG System

Hệ thống hỏi đáp pháp luật Việt Nam sử dụng Retrieval-Augmented Generation (RAG) với Qdrant (hybrid search) và Google Gemini.

</div>

## Tổng quan

Dự án xây dựng pipeline RAG cho bài toán hỏi đáp pháp luật tiếng Việt:
- Truy xuất văn bản pháp luật liên quan bằng dense/sparse/hybrid search.
- Tổng hợp câu trả lời có căn cứ pháp lý từ ngữ cảnh truy xuất.
- Hỗ trợ cả CLI (`main.py`) và giao diện chat Streamlit (`streamlit_app.py`).

## Tính năng chính

- Hybrid retrieval trên Qdrant: `dense` + `sparse` (BM25/FastEmbed).
- Pipeline sinh câu trả lời bằng LangChain + Gemini (`src/generation/rag.py`).
- Index dữ liệu pháp luật vào Qdrant từ corpus JSON (`src/indexing/qdrantvdb.py`).
- Giao diện chat Streamlit có hiển thị nguồn tham khảo.
- Script đánh giá retrieval đa mức top-k (`src/eval/eval_retrieve.py`).

## Công nghệ

- Python 3.13+
- Qdrant
- LangChain
- Google Gemini (`langchain-google-genai`)
- sentence-transformers
- fastembed
- Streamlit

## Cấu trúc thư mục

```text
.
├── config/
│   └── config.py
├── src/
│   ├── data/
│   │   ├── document_loader.py
│   │   ├── legal_splitter.py
│   │   └── unit_merger.py
│   ├── embedding/
│   │   └── embedd_model.py
│   ├── eval/
│   │   └── eval_retrieve.py
│   ├── generation/
│   │   └── rag.py
│   ├── indexing/
│   │   └── qdrantvdb.py
│   ├── retrieve/
│   │   └── retrieve_rerank.py
│   └── vectorstores/
│       └── qdrant_store.py
├── main.py
├── streamlit_app.py
├── pyproject.toml
└── .env.example
```

## Cài đặt nhanh

### 1. Tạo môi trường và cài dependencies

```bash
uv sync
```

Hoặc dùng `pip`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### 2. Cấu hình biến môi trường

```bash
copy .env.example .env
```

Điền các giá trị cần thiết trong `.env`, đặc biệt là `GOOGLE_API_KEY`.



## Cấu hình `.env`

Các biến chính (theo `.env.example`):

- `QDRANT_URL` (mặc định: `http://localhost:6333`)
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION_NAME` (mặc định: `legal_documents`)
- `EMBEDDING_MODEL_NAME` (mặc định: `AITeamVN/Vietnamese_Embedding_v2`)
- `SPARSE_EMBEDDING_MODEL_NAME` (mặc định: `Qdrant/bm25`)
- `EMBEDDING_SIZE` (mặc định: `1024`)
- `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `DEFAULT_TOP_K`, `BATCH_SIZE`, `UPSERT_BATCH_SIZE`
- `LLM_PROVIDER` (mặc định: `gemini`)
- `LLM_MODEL` (ví dụ: `gemini-1.5-flash`)
- `LLM_TEMPERATURE`
- `GOOGLE_API_KEY`

## Cách chạy

### Chạy chat UI (Streamlit)

```bash
streamlit run streamlit_app.py
```

### Chạy CLI

```bash
python main.py
```

`main.py` đang để sẵn luồng query mẫu. Nếu muốn index dữ liệu mới, bật lại phần `OPTION 1` trong file này.

## Đánh giá retrieval

Chạy script đánh giá với tập test JSON:

```bash
python -m src.eval.eval_retrieve --test_file src/data/test_data.json --output evaluation_results.json
```

Có thể thêm các tham số như `--limit`, `--workers`, `--beta`.

## Lưu ý

- Cần có collection và dữ liệu đã được index trong Qdrant trước khi hỏi đáp.
- Nếu dùng model embedding lớn, nên chạy máy có GPU để tăng tốc độ index/evaluate.
- Nếu gặp lỗi import ở `main.py` liên quan `new_qdrantvdb`, hãy đổi về `from src.indexing.qdrantvdb import QdrantVDB` cho đúng cấu trúc hiện tại.

## License

MIT License.
