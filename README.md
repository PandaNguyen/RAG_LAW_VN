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

## Kết quả đánh giá retriever



Kết quả đầy đủ đa mức top-k

- `num_queries_evaluated`: `2065`
- `top_k_list`: `1, 3, 5, 10, 20`
- `beta`: `2.0`

#### Dense Retriever

| Top-k | Recall | Precision | F2.0 | MRR |
|---|---:|---:|---:|---:|
| 1 | 0.6652 | 0.7651 | 0.6750 | 0.7651 |
| 3 | 0.8144 | 0.3225 | 0.6084 | 0.8290 |
| 5 | 0.8482 | 0.2053 | 0.5053 | 0.8348 |
| 10 | 0.8829 | 0.1092 | 0.3530 | 0.8373 |
| 20 | 0.9111 | 0.0574 | 0.2226 | 0.8384 |

#### Sparse Retriever

| Top-k | Recall | Precision | F2.0 | MRR |
|---|---:|---:|---:|---:|
| 1 | 0.3806 | 0.4383 | 0.3863 | 0.4383 |
| 3 | 0.5518 | 0.2181 | 0.4118 | 0.5205 |
| 5 | 0.6180 | 0.1483 | 0.3670 | 0.5360 |
| 10 | 0.6943 | 0.0848 | 0.2758 | 0.5456 |
| 20 | 0.7566 | 0.0467 | 0.1823 | 0.5496 |

#### Hybrid Retriever

| Top-k | Recall | Precision | F2.0 | MRR |
|---|---:|---:|---:|---:|
| 1 | 0.5869 | 0.6765 | 0.5957 | 0.6765 |
| 3 | 0.7577 | 0.3001 | 0.5659 | 0.7517 |
| 5 | 0.8058 | 0.1946 | 0.4796 | 0.7615 |
| 10 | 0.8649 | 0.1062 | 0.3443 | 0.7683 |
| 20 | 0.9029 | 0.0566 | 0.2196 | 0.7699 |

Nhận xét nhanh:
- Dense đang cho hiệu năng tốt nhất theo F2.0 và MRR trên hầu hết các mức top-k.
- Hybrid bám sát dense ở recall khi tăng top-k, nhưng precision thấp dần theo top-k (xu hướng chung của retrieval).
- Sparse có hiệu năng thấp hơn dense/hybrid trên bộ dữ liệu hiện tại.

## Lưu ý

- Cần có collection và dữ liệu đã được index trong Qdrant trước khi hỏi đáp.
- Nếu dùng model embedding lớn, nên chạy máy có GPU để tăng tốc độ index/evaluate.
- Nếu gặp lỗi import ở `main.py` liên quan `new_qdrantvdb`, hãy đổi về `from src.indexing.qdrantvdb import QdrantVDB` cho đúng cấu trúc hiện tại.

## License

MIT License.
