"""选型加载、固定 Corpus、可重载 FAISS 与真实组件装配。"""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from apt_rag.evaluation.embedding_benchmark import _write_json
from apt_rag.evaluation.retrieval_benchmark import prepare_inputs
from apt_rag.generation.providers import build_provider, fingerprint
from apt_rag.rag.budget import TokenBudget
from apt_rag.rag.pipeline import RAGPipeline
from apt_rag.retrieval import BM25Retriever, RankedChunk
from apt_rag.stack_selection import validate_selection


class PreparationOnlyLLM:
    """云端 dry-run 不读取密钥或访问 API；误用为生成时明确拒绝。"""

    def __init__(self, spec):
        self.spec, self.client = spec, None

    def identity(self):
        raise ValueError("仅预检实例不允许生成；请显式允许云端后重新构建")

    def generate(self, messages):
        raise ValueError("仅预检实例不允许生成")


def load_settings(root: Path, config_path: Path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["schema_version"] != "1.0":
        raise ValueError("不支持的 RAG 配置版本")
    decision = json.loads((root / config["stack_path"]).read_text(encoding="utf-8"))
    sources = validate_selection(root, decision)
    selected = decision["selection"]
    if selected["vectorstore_backend"] != "faiss" or selected["retrieval_strategy"] != "hybrid_reranker":
        raise ValueError("Phase 9 只集成已选择的 FAISS + Hybrid + Reranker")
    for key in ("context_window_tokens", "token_safety_margin", "max_question_characters"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError("长度配置必须为正整数")
    providers = {p["id"]: p for p in sources["generation"]["providers"]}
    local = providers[selected["local_llm_provider_id"]]
    if (config["context_window_tokens"] > local["num_ctx"] or
            config["context_window_tokens"] > config["tokenizers"][selected["primary_llm_provider_id"]]["documented_context_tokens"] or
            config["token_safety_margin"] + sources["generation"]["max_output_tokens"] >= config["context_window_tokens"]):
        raise ValueError("共享 Context 预算超出服务窗口或没有输入空间")
    for path in (config["index_directory"], config["tokenizers"][selected["primary_llm_provider_id"]]["directory"]):
        if not (root / path).resolve().is_relative_to(root.resolve() / "data/processed"):
            raise ValueError("生成缓存必须保存在 data/processed 内")
    return config, decision, sources


class FaissRetriever:
    def __init__(self, store, corpus_ids):
        self.store, self.ids = store, corpus_ids
        self.positions = {key: index for index, key in enumerate(corpus_ids)}

    def retrieve(self, query, query_vector, top_k):
        # 多取一个检查边界同分；同分时枚举该小规模索引，不任意丢弃并列项。
        hits = self.store.search(query_vector, min(top_k + 1, len(self.ids)))
        if len(hits) > top_k and hits[top_k - 1].score == hits[top_k].score:
            hits = self.store.search(query_vector, len(self.ids))
        return [RankedChunk(h.chunk_id, h.score) for h in sorted(
            hits, key=lambda h: (-h.score, self.positions[h.chunk_id]),
        )[:top_k]]


def fixed_inputs(root, sources):
    return prepare_inputs(root, sources["retrieval"])


def index_header(config, decision, inputs):
    return {"rag_config_fingerprint": fingerprint(config), "decision_fingerprint": fingerprint(decision),
            "corpus_fingerprint": inputs["controls"]["corpus_fingerprint"],
            "corpus_vectors_sha256": inputs["controls"]["corpus_vectors_sha256"], "ids": inputs["ids"]}


def open_index(root, config, decision, sources, inputs, *, create=False):
    from apt_rag.vectorstore.local import FaissStore

    directory = root / config["index_directory"]
    header = index_header(config, decision, inputs)
    fields = sources["phase5"]["controlled_variables"]["metadata_fields"]
    metadata = [{key: chunk[key] for key in fields} for chunk in inputs["chunks"]]
    if not directory.exists():
        if not create:
            raise ValueError("RAG 索引未准备；请先运行 scripts/prepare_rag.py")
        store = FaissStore(directory, sources["vectorstore"]["hnsw"])
        store.build(inputs["vectors"], inputs["ids"], metadata)
        _write_json(directory / "audit.json", {**header, "files": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("index.faiss", "metadata.json")}, "top30_verified": False})
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    if any(audit.get(key) != value for key, value in header.items()):
        raise ValueError("索引身份/配置改变；请用新的缓存目录，不覆盖旧数据")
    if any(hashlib.sha256((directory / name).read_bytes()).hexdigest() != value for name, value in audit["files"].items()):
        raise ValueError("索引/Metadata 文件损坏或被改动")
    store = FaissStore.load(directory)
    if store.ids != inputs["ids"] or store.metadata != metadata or store.hnsw != sources["vectorstore"]["hnsw"]:
        raise ValueError("索引 ID/Metadata/HNSW 不匹配")
    if not create and not audit["top30_verified"]:
        raise ValueError("索引尚未通过 Top-30 接入验证")
    return store, audit


def build_pipeline(root, config_path, provider_id=None, *, allow_cloud=False, dry_run=False):
    config, decision, sources = load_settings(root, config_path)
    allowed = {decision["selection"][k] for k in ("primary_llm_provider_id", "local_llm_provider_id")}
    provider_id = provider_id or decision["selection"]["primary_llm_provider_id"]
    if provider_id not in allowed:
        raise ValueError("只能显式选择 Phase 8 选定的生成后端")
    if provider_id == decision["selection"]["primary_llm_provider_id"] and not allow_cloud and not dry_run:
        raise ValueError("云端生成必须显式 allow_cloud=True，不自动发送问题或证据")
    inputs = fixed_inputs(root, sources)
    store, _ = open_index(root, config, decision, sources, inputs)
    from apt_rag.embedding import SentenceTransformerProvider
    from apt_rag.reranking.bge import BGEReranker

    # 先固定 Torch 线程，再加载两个 CPU 模型；不改变旧依赖/GPU 安装。
    reranker = BGEReranker(sources["retrieval"]["reranker"])
    model = next(m for m in sources["embedding"]["models"] if m["id"] == decision["selection"]["embedding_model_id"])
    embedding = SentenceTransformerProvider(model["model_name"], revision=model["revision"],
        device=sources["embedding"]["device"], batch_size=sources["embedding"]["batch_size"],
        max_seq_length=sources["embedding"]["max_seq_length"], local_files_only=True, reject_truncated_inputs=True)
    spec = next(p for p in sources["generation"]["providers"] if p["id"] == provider_id)
    settings = {**sources["generation"], "strict_context": True}
    llm = PreparationOnlyLLM(spec) if dry_run and spec["kind"] == "deepseek" else build_provider(spec, settings)
    try:
        budget = TokenBudget(root, config, spec, settings["max_output_tokens"], provider=llm)
    except Exception:
        if llm.client is not None:
            llm.client.close()
        raise
    sparse = BM25Retriever([c["text"] for c in inputs["chunks"]], inputs["ids"],
                          k1=sources["retrieval"]["bm25"]["k1"], b=sources["retrieval"]["bm25"]["b"])
    pipeline = RAGPipeline(chunks=inputs["chunks"], embedding=embedding,
        dense=FaissRetriever(store, inputs["ids"]), sparse=sparse, reranker=reranker,
        llm=llm, budget=budget, prompt=sources["prompt"], retrieval_config=sources["retrieval"],
        top_k=decision["policies"]["context_top_k"], max_question_characters=config["max_question_characters"],
        public_corpus_verified=True)
    return pipeline, {"rag_config_fingerprint": fingerprint(config), "decision_id": decision["decision_id"],
                      "provider_spec": spec, "corpus_fingerprint": inputs["controls"]["corpus_fingerprint"],
                      "context_mode": "retrieved_chunks", "runtime": {"python": platform.python_version(),
                          "platform": platform.platform(), "packages": {name: importlib.metadata.version(name)
                              for name in ("numpy", "faiss-cpu", "torch", "transformers", "tokenizers", "httpx")}}}
