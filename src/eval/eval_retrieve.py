import json
import argparse
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
from fastembed import SparseTextEmbedding
from src.embedding.embedd_model import VietnameseLegalEmbedding
import torch
from tqdm import tqdm
from config.config import settings


def calculate_metrics(retrieved_ids, ground_truth, beta=2.0):
    if not ground_truth:
        return {"recall": 0.0, "precision": 0.0, f"f{beta}_score": 0.0, "mrr": 0.0}
    tp = len(set(retrieved_ids) & set(ground_truth))
    recall = tp / len(ground_truth)
    precision = tp / len(retrieved_ids) if retrieved_ids else 0.0
    f_beta = 0.0
    if recall + precision > 0:
        f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    mrr = next((1.0 / (i + 1) for i, r in enumerate(retrieved_ids) if r in ground_truth), 0.0)
    return {"recall": recall, "precision": precision, f"f{beta}_score": f_beta, "mrr": mrr}


def extract_ids(hits):
    return [
        f"{h.payload['metadata'].get('doc_number')}_{h.payload['metadata'].get('unit_id')}"
        for h in hits
    ]


class RetrievalEvaluator:
    def __init__(self, collection_name=None, max_workers=8):
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.max_workers = max_workers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading embedding models on {self.device}...")
        self.embedding_model = VietnameseLegalEmbedding(model_name=settings.embedding_model_name)
        self.sparse_embedding = SparseTextEmbedding(model_name=settings.sparse_embedding_model_name)

        print("Connecting to Qdrant...")
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    # ------------------------------------------------------------------ #
    #  Embed toàn bộ queries một lần (batch)                              #
    # ------------------------------------------------------------------ #
    def _batch_dense(self, texts: List[str]) -> List[List[float]]:
        """Embed nhiều texts cùng lúc."""
        return self.embedding_model.embed_documents(texts, batch_size=16)   # hoặc encode_batch

    def _batch_sparse(self, texts: List[str]) -> List[SparseVector]:
        results = []
        for s in self.sparse_embedding.embed(texts):         # fastembed hỗ trợ batch
            results.append(SparseVector(indices=s.indices.tolist(), values=s.values.tolist()))
        return results

    # ------------------------------------------------------------------ #
    #  Gọi 3 strategies song song cho 1 query                            #
    # ------------------------------------------------------------------ #
    def _search_one_query(self, dense_vec, sparse_vec, top_k) -> Dict[str, List[str]]:
        """Gửi 3 requests Qdrant song song bằng ThreadPoolExecutor."""

        def dense_fn():
            return extract_ids(self.client.query_points(
                collection_name=self.collection_name,
                query=dense_vec, using="dense",
                with_payload=True, limit=top_k,
            ).points)

        def sparse_fn():
            return extract_ids(self.client.query_points(
                collection_name=self.collection_name,
                query=sparse_vec, using="sparse",
                with_payload=True, limit=top_k,
            ).points)

        def hybrid_fn():
            return extract_ids(self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(query=dense_vec, using="dense", limit=top_k),
                    Prefetch(query=sparse_vec, using="sparse", limit=top_k),
                ],
                query=FusionQuery(fusion=Fusion.RRK),
                with_payload=True, limit=top_k,
            ).points)

        with ThreadPoolExecutor(max_workers=3) as ex:
            fd, fs, fh = ex.submit(dense_fn), ex.submit(sparse_fn), ex.submit(hybrid_fn)
            return {"dense": fd.result(), "sparse": fs.result(), "hybrid": fh.result()}

    # ------------------------------------------------------------------ #
    #  Evaluate                                                           #
    # ------------------------------------------------------------------ #
    def evaluate(self, test_data: List[Dict], top_k: int = 5, beta: float = 2.0):
        # Lọc queries hợp lệ trước
        valid = []
        for item in test_data:
            gt = list({
                f"{ru.get('law_id','')}_{ru.get('unit_id_str','')}"
                for ru in item.get("relevant_law_units", [])
            })
            if gt:
                valid.append((item["question"], gt))

        queries  = [q for q, _ in valid]
        gt_list  = [g for _, g in valid]

        # ① Batch embed TẤT CẢ queries cùng lúc
        print("Batch embedding queries...")
        dense_vecs  = self._batch_dense(queries)
        sparse_vecs = self._batch_sparse(queries)

        # ② Gọi Qdrant song song theo queries (outer) + song song 3 strategies (inner)
        metrics = {s: {"recall": 0.0, "precision": 0.0, f"f{beta}_score": 0.0, "mrr": 0.0}
                   for s in ("dense", "sparse", "hybrid")}

        def process(i):
            res = self._search_one_query(dense_vecs[i], sparse_vecs[i], top_k)
            return {
                s: calculate_metrics(res[s], gt_list[i], beta)
                for s in ("dense", "sparse", "hybrid")
            }

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(process, i): i for i in range(len(valid))}
            for fut in tqdm(as_completed(futures), total=len(valid), desc="Evaluating"):
                m = fut.result()
                for s in metrics:
                    for k, v in m[s].items():
                        metrics[s][k] += v

        n = len(valid)
        for s in metrics:
            for k in metrics[s]:
                metrics[s][k] /= n

        return {"num_queries_evaluated": n, "top_k": top_k, "beta": beta, "results": metrics}
    def evaluate_multi_k(
        self,
        test_data: List[Dict],
        top_k_list: List[int] = [1, 3, 5, 10, 20],
        beta: float = 2.0
    ):
        """Search 1 lần với max(top_k_list), evaluate cho tất cả k."""
        
        top_k_max = max(top_k_list)

        # Lọc valid queries
        valid = []
        for item in test_data:
            gt = list({
                f"{ru.get('law_id','')}_{ru.get('unit_id_str','')}"
                for ru in item.get("relevant_law_units", [])
            })
            if gt:
                valid.append((item["question"], gt))

        queries = [q for q, _ in valid]
        gt_list = [g for _, g in valid]

        # Batch embed 1 lần duy nhất
        print("Batch embedding queries...")
        batch_size = 64 if self.device == "cuda" else 16
        dense_vecs  = self.embedding_model.embed_documents(queries, batch_size=batch_size)
        sparse_vecs = self._batch_sparse(queries)

        # Init metrics cho tất cả (strategy, k)
        strategies = ("dense", "sparse", "hybrid")
        metrics = {
            k: {s: {"recall": 0.0, "precision": 0.0, f"f{beta}_score": 0.0, "mrr": 0.0}
                for s in strategies}
            for k in top_k_list
        }

        def process(i):
            # Search 1 lần với top_k_max
            full_results = self._search_one_query(dense_vecs[i], sparse_vecs[i], top_k_max)
            
            # Slice và tính metrics cho từng k
            result = {}
            for k in top_k_list:
                result[k] = {
                    s: calculate_metrics(full_results[s][:k], gt_list[i], beta)
                    for s in strategies
                }
            return result

        # Chạy song song
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(process, i): i for i in range(len(valid))}
            for fut in tqdm(as_completed(futures), total=len(valid), desc="Evaluating"):
                m = fut.result()
                for k in top_k_list:
                    for s in strategies:
                        for metric, val in m[k][s].items():
                            metrics[k][s][metric] += val

        # Average
        n = len(valid)
        for k in top_k_list:
            for s in strategies:
                for metric in metrics[k][s]:
                    metrics[k][s][metric] /= n

        return {
            "num_queries_evaluated": n,
            "top_k_list": top_k_list,
            "top_k_max_searched": top_k_max,
            "beta": beta,
            "results": metrics,
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--top_k",     type=int,   default=5)
    parser.add_argument("--beta",      type=float, default=2.0)
    parser.add_argument("--limit",     type=int,   default=None)
    parser.add_argument("--workers",   type=int,   default=8)
    parser.add_argument("--output",    default="evaluation_results2.json")
    args = parser.parse_args()
    print("test start")
    with open(args.test_file, encoding="utf-8") as f:
        test_data = json.load(f)
    if args.limit:
        test_data = test_data[:args.limit]

    evaluator = RetrievalEvaluator(max_workers=args.workers)
    results   = evaluator.evaluate_multi_k(test_data, top_k_list=[1, 3, 5, 10, 20], beta=args.beta)

    print("\n" + "="*50)
    print(json.dumps(results, indent=4, ensure_ascii=False))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
