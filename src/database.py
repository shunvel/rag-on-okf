from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import Config
from src.parsers import OKFParser


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class KnowledgeStore:
    def __init__(self) -> None:
        self.client = QdrantClient(
            url=f"http://{Config.QDRANT_HOST}:{Config.QDRANT_PORT}",
            check_compatibility=False,
        )
        self.encoder = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.collection_name = Config.COLLECTION_NAME
        self.bm25: BM25Okapi | None = None
        self.raw_documents: list[dict[str, Any]] = []
        self.docs_by_id: dict[int, dict[str, Any]] = {}

    def healthcheck(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except (UnexpectedResponse, OSError, ConnectionError):
            return False

    def collection_point_count(self) -> int:
        if not self.client.collection_exists(self.collection_name):
            return 0
        info = self.client.get_collection(self.collection_name)
        return int(info.points_count or 0)

    def ensure_ready(self, force_reseed: bool = False) -> None:
        exists = self.client.collection_exists(self.collection_name)
        count = self.collection_point_count() if exists else 0
        if force_reseed and exists:
            self.client.delete_collection(self.collection_name)
            exists = False
            count = 0
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=Config.VECTOR_SIZE, distance=Distance.COSINE),
            )
        if count == 0:
            self.seed_data()
        else:
            self._hydrate_from_qdrant()

    def seed_data(self) -> None:
        self.raw_documents = self._build_corpus()
        self._rebuild_bm25()
        points = []
        for doc in self.raw_documents:
            vector = self.encoder.encode(doc["text"]).tolist()
            points.append(
                PointStruct(
                    id=doc["id"],
                    vector=vector,
                    payload={
                        "type": doc["type"],
                        "text": doc["text"],
                        "path": doc["path"],
                        "aliases": doc["aliases"],
                        "tags": doc["tags"],
                        "title": doc["title"],
                    },
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)
        self.docs_by_id = {doc["id"]: doc for doc in self.raw_documents}

    def encode_query(self, query: str) -> list[float]:
        return self.encoder.encode(query).tolist()

    def all_points_with_vectors(self) -> list[dict[str, Any]]:
        if not self.client.collection_exists(self.collection_name):
            return []
        points: list[dict[str, Any]] = []
        offset = None
        while True:
            batch, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=128,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in batch:
                payload = point.payload or {}
                vector = point.vector
                if isinstance(vector, dict):
                    vector = next(iter(vector.values()), [])
                points.append(
                    {
                        "id": point.id,
                        "vector": list(vector or []),
                        "type": payload.get("type", ""),
                        "title": payload.get("title", ""),
                        "text": payload.get("text", ""),
                    }
                )
            if offset is None:
                break
        return points

    def hybrid_search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        ranked = self.hybrid_scores(query)
        return ranked[: (top_k or Config.TOP_K)]

    def hybrid_scores(self, query: str) -> list[dict[str, Any]]:
        """Dense cosine + BM25, merged with Reciprocal Rank Fusion (not a 60/40 score mix)."""
        if not self.raw_documents or self.bm25 is None:
            self._hydrate_from_qdrant()
        query_vector = self.encode_query(query)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=max(len(self.raw_documents), 1),
            with_payload=True,
        )
        tokenized_query = _tokenize(query)
        bm25_raw = self.bm25.get_scores(tokenized_query)
        bm25_max = float(max(bm25_raw) + 1e-5)

        dense_by_id: dict[int, float] = {}
        for point in response.points:
            doc_id = int(point.id)
            if doc_id in self.docs_by_id:
                dense_by_id[doc_id] = max(0.0, float(point.score))

        bm25_by_id: dict[int, float] = {}
        for doc in self.raw_documents:
            raw = float(bm25_raw[doc["corpus_index"]])
            bm25_by_id[doc["id"]] = raw / bm25_max

        dense_rank = {
            doc_id: rank
            for rank, doc_id in enumerate(sorted(dense_by_id, key=dense_by_id.get, reverse=True), start=1)
        }
        bm25_rank = {
            doc_id: rank
            for rank, doc_id in enumerate(sorted(bm25_by_id, key=bm25_by_id.get, reverse=True), start=1)
        }

        scored = []
        for doc in self.raw_documents:
            doc_id = doc["id"]
            rrf = (1.0 / (Config.RRF_K + dense_rank[doc_id])) + (1.0 / (Config.RRF_K + bm25_rank[doc_id]))
            scored.append(
                {
                    **doc,
                    "dense_score": dense_by_id.get(doc_id, 0.0),
                    "bm25_score": bm25_by_id.get(doc_id, 0.0),
                    "dense_rank": dense_rank[doc_id],
                    "bm25_rank": bm25_rank[doc_id],
                    "rrf_score": rrf,
                    "fused_score": rrf,
                }
            )
        scored.sort(key=lambda item: item["rrf_score"], reverse=True)
        return scored

    def _hydrate_from_qdrant(self) -> None:
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        docs = []
        for index, point in enumerate(records):
            payload = point.payload or {}
            docs.append(
                {
                    "id": int(point.id),
                    "corpus_index": index,
                    "type": payload.get("type", "rag_unstructured"),
                    "text": payload.get("text", ""),
                    "path": payload.get("path", ""),
                    "aliases": payload.get("aliases") or [],
                    "tags": payload.get("tags") or [],
                    "title": payload.get("title", ""),
                }
            )
        docs.sort(key=lambda item: item["id"])
        for index, doc in enumerate(docs):
            doc["corpus_index"] = index
        self.raw_documents = docs
        self.docs_by_id = {doc["id"]: doc for doc in docs}
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        tokenized = [_tokenize(doc["text"]) for doc in self.raw_documents]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def _build_corpus(self) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        next_id = 1

        for concept in OKFParser.load_concept_files(Config.OKF_BUNDLE_DIR):
            meta = concept["metadata"]
            aliases = [str(item).lower() for item in (meta.get("aliases") or [])]
            tags = [str(item).lower() for item in (meta.get("tags") or [])]
            title = str(meta.get("title") or "")
            search_text = " ".join(
                part for part in [title, " ".join(aliases), " ".join(tags), concept.get("body", "")] if part
            )
            documents.append(
                {
                    "id": next_id,
                    "type": "okf_concept",
                    "text": search_text,
                    "path": concept["path"],
                    "aliases": aliases,
                    "tags": tags,
                    "title": title,
                }
            )
            next_id += 1

        noise_path = os.path.join(Config.DATA_DIR, "unstructured_noise.json")
        with open(noise_path, encoding="utf-8") as handle:
            for item in json.load(handle):
                documents.append(
                    {
                        "id": next_id,
                        "type": "rag_unstructured",
                        "text": item["text"],
                        "path": "",
                        "aliases": [],
                        "tags": [],
                        "title": f"noise-{item.get('id', next_id)}",
                    }
                )
                next_id += 1

        csv_path = os.path.join(Config.DATA_DIR, "saas_financial_market.csv")
        frame = pd.read_csv(csv_path)
        sample = frame.groupby("company_id", as_index=False).tail(1)
        for _, row in sample.iterrows():
            text = (
                f"Warehouse snapshot for company {row['company_id']} in {row['period']}: "
                f"MRR_USD={row['MRR_USD']}, Churn_Rate={row['Churn_Rate']}, "
                f"ARPU_USD={row['ARPU_USD']}, CAC_USD={row['CAC_USD']}, "
                f"Customer_Count={row['Customer_Count']}. Operational values, not metric definitions."
            )
            documents.append(
                {
                    "id": next_id,
                    "type": "warehouse_fact",
                    "text": text,
                    "path": csv_path,
                    "aliases": [],
                    "tags": ["warehouse", "snapshot"],
                    "title": f"{row['company_id']} {row['period']}",
                }
            )
            next_id += 1

        qa_path = os.path.join(Config.DATA_DIR, "financial_qa.json")
        with open(qa_path, encoding="utf-8") as handle:
            for item in json.load(handle):
                if item.get("metric"):
                    continue
                documents.append(
                    {
                        "id": next_id,
                        "type": "rag_unstructured",
                        "text": f"Disclosure Q&A: {item['question']} {item['answer']}",
                        "path": "",
                        "aliases": [],
                        "tags": [],
                        "title": item["id"],
                    }
                )
                next_id += 1

        for index, doc in enumerate(documents):
            doc["corpus_index"] = index
        return documents
