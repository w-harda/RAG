from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from apt_rag.vectorstore.local import ChromaStore, FaissStore, load_store

HNSW = {"max_neighbors": 16, "ef_construction": 100, "ef_search": 100, "num_threads": 1}
PERSISTENCE = {"batch_size": 100, "sync_threshold": 100}


@pytest.mark.parametrize("backend", ["faiss", "chroma"])
def test_store_retrieval_filter_and_fresh_process_reload(tmp_path: Path, backend: str) -> None:
    path = tmp_path / backend
    vectors = np.asarray([[1, 0], [0.8, 0.6], [0, 1]], dtype=np.float32)
    metadata = [{"vendor": "CISA", "page": 1}, {"vendor": "Mandiant", "page": 2},
                {"vendor": "Mandiant", "page": 3}]
    store = FaissStore(path, HNSW) if backend == "faiss" else ChromaStore(path, HNSW, PERSISTENCE)
    store.build(vectors, ["a", "b", "c"], metadata)
    hits = store.search(vectors[0], 10)
    assert [hit.chunk_id for hit in hits] == ["a", "b", "c"]
    assert [hit.score for hit in hits] == pytest.approx([1, 0.8, 0], abs=1e-6)
    assert hits[1].metadata == metadata[1]
    assert [hit.chunk_id for hit in store.search(vectors[0], 1, {"vendor": "Mandiant"})] == ["b"]
    assert [hit.chunk_id for hit in store.search(vectors[0], 10, {"vendor": "Mandiant", "page": 3})] == ["c"]
    assert store.search(vectors[0], 2, {"vendor": "missing"}) == []
    assert store.get("b") == metadata[1]
    with pytest.raises(KeyError):
        store.get("missing")

    # 真正另起解释器验证文件持久化，避免只验证进程内缓存。
    code = (
        "import json, sys, numpy as np; from pathlib import Path; "
        "from apt_rag.vectorstore.local import load_store; "
        "s=load_store(sys.argv[1],Path(sys.argv[2])); "
        "print(json.dumps({'ids':[h.chunk_id for h in s.search(np.array([1,0],dtype=np.float32),3)],"
        "'metadata':s.get('b')}))"
    )
    child = subprocess.run([sys.executable, "-c", code, backend, str(path)],
                           capture_output=True, text=True, check=True, timeout=60)
    assert json.loads(child.stdout) == {"ids": ["a", "b", "c"], "metadata": metadata[1]}
    restored = load_store(backend, path)
    assert restored.get("b") == metadata[1]


@pytest.mark.parametrize("backend", ["faiss", "chroma"])
def test_bad_vectors_and_existing_store_are_rejected(tmp_path: Path, backend: str) -> None:
    path = tmp_path / backend
    store = FaissStore(path, HNSW) if backend == "faiss" else ChromaStore(path, HNSW, PERSISTENCE)
    with pytest.raises(ValueError, match="归一化"):
        store.build(np.asarray([[2, 0]], dtype=np.float32), ["a"], [{"page": 1}])
    assert not path.exists()
    store.build(np.asarray([[1, 0]], dtype=np.float32), ["a"], [{"page": 1}])
    with pytest.raises(FileExistsError):
        store.build(np.asarray([[1, 0]], dtype=np.float32), ["a"], [{"page": 1}])
    with pytest.raises(ValueError, match="维数"):
        store.search(np.asarray([1, 0, 0], dtype=np.float32), 1)
