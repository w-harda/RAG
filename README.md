# APT-RAG

基于检索增强生成（RAG）的 APT 组织情报知识问答系统与对比验证工程。

项目以可复现、可比较、可解释为核心，按阶段构建 APT 报告语料库、检索 Benchmark、组件对比实验、完整 RAG Pipeline 和 Web UI。

## 当前进度

Phase 8：根据 Phase 4–7 的实验结果确定最终研究技术栈，记录依据与限制。

已完成 15 份 APT 报告、1,297 个可追溯 Chunk、20 题 Ground Truth Benchmark、三模型 Embedding 实测、FAISS/Chroma 对比、两个 LLM 的 40 份真实回答，以及三策略固定向量检索对比和 600 个真实 Reranker 分数。LLM 辅助评分尚未经独立人工评审；检索指标来自固定证据标注，不代表最终问答正确率。原始 PDF、处理文本、向量、模型和索引缓存不进入 Git，实验结果进入 Git。

## 环境要求

- Git
- uv
- Python 3.12（由 `.python-version` 和 `pyproject.toml` 约束）
- curl（仅在来源站点拒绝 Python 下载客户端时作为自动回退）

## 初始化

```powershell
uv sync --locked
```

`uv` 会创建本地 `.venv`，安装当前项目及开发依赖，并严格按照 `uv.lock` 解析环境。

## 验证

```powershell
uv run python --version
uv run pytest
uv run python scripts/validate_manifest.py
uv run python scripts/process_corpus.py
uv run python scripts/validate_processed.py
uv run python scripts/validate_benchmark.py
```

Python 版本应为 3.12，测试和 manifest 校验应全部通过。

## 获取报告语料

```powershell
uv run python scripts/download_reports.py
uv run python scripts/download_reports.py --verify-only
```

报告默认保存到 `data/raw/`，不会被 Git 跟踪。语料选择标准、字段说明和单份报告下载方式见 [语料文档](docs/corpus.md)。

PDF 解析、清洗、Chunk 配置和 Metadata 字段说明见 [处理流程文档](docs/processing.md)。

Benchmark 的设计、字段与证据相关性等级见 [Benchmark 文档](docs/benchmark.md)。

Embedding 控制变量、指标定义、运行命令和真实结果见 [Embedding 实验](docs/embedding_benchmark.md)。

FAISS/Chroma 的控制变量、建库速度、延迟、Metadata 与规模比较见 [向量存储实验](docs/vectorstore_benchmark.md)。已有 Embedding 缓存时运行：

```powershell
uv run python scripts/run_vectorstore_benchmark.py
uv run python scripts/plot_vectorstore_results.py
```

LLM 的控制变量、环境变量、真实结果和评审限制见 [LLM 实验](docs/llm_benchmark.md)。离线验证与绘图不需要密钥、Ollama 或模型：

```powershell
uv run python scripts/validate_llm_results.py
uv run python scripts/plot_llm_results.py
```

已有 Phase 2 Chunk 时，可仅重建输入，或运行模型实验（云端首次调用会计费）：

```powershell
uv run python scripts/run_llm_benchmark.py --prepare-only
uv run python scripts/run_llm_benchmark.py
```

已完成的回答默认复用，不再次调用模型。重新实测应使用独立配置及新结果目录，不覆盖本次记录。

检索控制变量、模型下载、真实指标和 CPU 开销见 [检索策略实验](docs/retrieval_benchmark.md)。离线验证/绘图无需模型或向量缓存：

```powershell
uv run python scripts/validate_retrieval_results.py
uv run python scripts/plot_retrieval_results.py
```

已有 Chunk 和向量时可检查完整检索输入，或运行实验（首次运行需先按文档下载固定 Reranker）：

```powershell
uv run python scripts/validate_retrieval_results.py --verify-inputs
uv run python scripts/run_retrieval_benchmark.py
```

已经完成的实验默认复用。

最终研究选型为 BGE-M3 + FAISS HNSW + Hybrid/RRF + BGE-Reranker-v2-m3 + DeepSeek Flash；本地 Ollama Qwen 保留为显式选择方案，不自动回退。该配置质量优先，CPU 重排单题约 31 秒，不承诺低延迟；组件选定不代表组合已经验证。完整依据、未验证边界和 Phase 9 验收要求见 [技术栈决策](docs/stack_selection.md)。

`configs/final_stack.json` 冻结组件选择与既有配置/结果的内容指纹，是选型记录而非运行管线。离线校验不需要密钥、模型或处理后语料：

```powershell
uv run python scripts/validate_stack_selection.py
```

Phase 8 未调用模型或付费 API，未改动旧实验数据；尚未实现 Phase 9 的完整 RAG Pipeline/Citation 或 Web UI。

## 当前目录

```text
.
├── src/apt_rag/       # Python 包
├── configs/           # 可复现实验配置
├── data/benchmark/    # 问题、标准答案与 Ground Truth Evidence
├── data/manifest/     # 报告清单与 JSON Schema
├── docs/              # 阶段文档
├── results/           # 已提交的真实实验结果与图表
├── scripts/           # 下载、处理与数据校验入口
├── tests/             # 自动化测试
├── .env.example       # 环境变量模板（不存放真实密钥）
├── .python-version    # Python 3.12 约束
├── pyproject.toml     # 项目与依赖配置
└── uv.lock            # 依赖锁文件（运行 uv sync 后生成）
```

原始报告文件默认由 `.gitignore` 排除。项目使用 manifest、官方来源 URL、文件大小和 SHA-256 校验值保证数据可追溯，并在来源文件变化时显式报错。
