"""FAISS 和 Chroma 的持久化 HNSW 适配器。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
import faiss
import numpy as np
from chromadb.config import Settings

from apt_rag.vectorstore.base import SearchHit, prepare_query, validate_batch


class FaissStore:
    def __init__(self, directory: Path, hnsw: dict[str, int]) -> None:
        self.directory = directory
        self.hnsw = hnsw
        faiss.omp_set_num_threads(hnsw["num_threads"])

    def build(self, vectors: np.ndarray, ids: list[str], metadata: list[dict[str, Any]]) -> None:
        validate_batch(vectors, ids, metadata)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.ids = list(ids)
        self.metadata = list(metadata)
        self.positions = {chunk_id: index for index, chunk_id in enumerate(ids)}
        self.index = faiss.IndexHNSWFlat(vectors.shape[1], self.hnsw["max_neighbors"], faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.hnsw["ef_construction"]
        self.index.hnsw.efSearch = self.hnsw["ef_search"]
        self.index.add(np.ascontiguousarray(vectors))
        faiss.write_index(self.index, str(self.directory / "index.faiss"))
        (self.directory / "metadata.json").write_text(
            json.dumps({"ids": ids, "metadata": metadata, "hnsw": self.hnsw}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "FaissStore":
        data = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        store = cls(directory, data["hnsw"])
        store.ids, store.metadata = data["ids"], data["metadata"]
        store.positions = {key: index for index, key in enumerate(store.ids)}
        store.index = faiss.read_index(str(directory / "index.faiss"))
        if store.index.ntotal != len(store.ids):
            raise ValueError("FAISS 索引与 Metadata 计数不一致")
        return store

    def search(self, vector: np.ndarray, top_k: int,
               where: dict[str, Any] | None = None) -> list[SearchHit]:
        query = prepare_query(vector, self.index.d, top_k)
        if where:
            # FAISS 不原生管理 Metadata：在完整匹配集合中精确搜索，避免先 Top-K 再过滤漏召回。
            positions = [i for i, record in enumerate(self.metadata)
                         if all(record.get(key) == value for key, value in where.items())]
            if not positions:
                return []
            filtered = faiss.IndexFlatIP(self.index.d)
            filtered.add(np.stack([self.index.reconstruct(i) for i in positions]))
            scores, indexes = filtered.search(query, min(top_k, len(positions)))
            pairs = [(float(score), positions[int(i)]) for score, i in zip(scores[0], indexes[0], strict=True)]
        else:
            scores, indexes = self.index.search(query, min(top_k, len(self.ids)))
            pairs = [(float(score), int(i)) for score, i in zip(scores[0], indexes[0], strict=True) if i >= 0]
        return [SearchHit(self.ids[i], score, self.metadata[i]) for score, i in pairs]

    def get(self, chunk_id: str) -> dict[str, Any]:
        return self.metadata[self.positions[chunk_id]]


class ChromaStore:
    def __init__(self, directory: Path, hnsw: dict[str, int], persistence: dict[str, int]) -> None:
        self.directory = directory
        self.hnsw = hnsw
        self.persistence = persistence

    def build(self, vectors: np.ndarray, ids: list[str], metadata: list[dict[str, Any]]) -> None:
        validate_batch(vectors, ids, metadata)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.client = chromadb.PersistentClient(str(self.directory), settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.create_collection(
            name="apt-chunks", embedding_function=None,
            configuration={"hnsw": {"space": "ip", **self.hnsw, **self.persistence}},
        )
        self.dimension = vectors.shape[1]
        self.count = len(ids)
        for start in range(0, len(ids), self.persistence["batch_size"]):
            end = start + self.persistence["batch_size"]
            self.collection.add(ids=ids[start:end], embeddings=vectors[start:end], metadatas=metadata[start:end])
        if self.collection.count() != len(ids):
            raise ValueError("Chroma 未完整写入向量")
        (self.directory / "settings.json").write_text(
            json.dumps({"dimension": self.dimension, "count": self.count}), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "ChromaStore":
        data = json.loads((directory / "settings.json").read_text(encoding="utf-8"))
        store = cls(directory, {}, {})
        store.client = chromadb.PersistentClient(str(directory), settings=Settings(anonymized_telemetry=False))
        store.collection = store.client.get_collection("apt-chunks", embedding_function=None)
        store.dimension, store.count = data["dimension"], data["count"]
        if store.collection.count() != store.count:
            raise ValueError("Chroma 持久化计数不一致")
        return store

    def search(self, vector: np.ndarray, top_k: int,
               where: dict[str, Any] | None = None) -> list[SearchHit]:
        query = prepare_query(vector, self.dimension, top_k)
        predicate = None
        if where:
            clauses = [{key: {"$eq": value}} for key, value in where.items()]
            predicate = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        result = self.collection.query(query_embeddings=query, n_results=min(top_k, self.count),
                                       where=predicate, include=["distances", "metadatas"])
        return [SearchHit(key, 1.0 - float(distance), metadata)
                for key, distance, metadata in zip(result["ids"][0], result["distances"][0],
                                                    result["metadatas"][0], strict=True)]

    def get(self, chunk_id: str) -> dict[str, Any]:
        result = self.collection.get(ids=[chunk_id], include=["metadatas"])
        if not result["ids"]:
            raise KeyError(chunk_id)
        return result["metadatas"][0]


def load_store(backend: str, directory: Path) -> FaissStore | ChromaStore:
    if backend == "faiss":
        return FaissStore.load(directory)
    if backend == "chroma":
        return ChromaStore.load(directory)
    raise ValueError(f"未知后端：{backend}")
