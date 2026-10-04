# APT-RAG

基于检索增强生成（RAG）的 APT 组织情报知识问答系统与对比验证工程。

项目以可复现、可比较、可解释为核心，按阶段构建 APT 报告语料库、检索 Benchmark、组件对比实验、完整 RAG Pipeline 和 Web UI。

## 当前进度

Phase 11：提供 Streamlit 聊天式前端，直接复用 Phase 9 管线；Phase 10 的独立人工质量复核仍待完成。

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

Phase 8 未调用模型或付费 API，未改动旧实验数据。

Phase 9 已验证 20 题 FAISS Top-30 与精确参考/Phase 7 候选的一致性、两后端完整 Context 的长度预算，并保存两条真实端到端冒烟。问答从检索 Chunk 构造 Context，引用的标题、页码、章节、URL 和 Chunk ID 来自 Metadata，不由模型生成。完整下载/准备步骤、预算与引用限制见 [RAG 管线说明](docs/rag_pipeline.md)。

```powershell
# 输入/模型准备步骤先按管线文档执行。
uv run python scripts/prepare_rag.py --check-budgets
uv run python scripts/ask_rag.py "APT44 通常还被称为什么？它被归属于俄罗斯哪个军事情报单位，其行动范围有何特点？" --provider ollama-qwen
# 云端显式授权，问题与公开片段会发送给 DeepSeek，生成会计费。
uv run python scripts/ask_rag.py "APT44 通常还被称为什么？它被归属于俄罗斯哪个军事情报单位，其行动范围有何特点？" --allow-cloud
# 离线检查，不要求模型/密钥/服务。
uv run python scripts/validate_rag_results.py
```

引用映射通过不等于回答语义正确。Phase 10 已新增固定同一 LLM/Prompt/问题集的 Pure LLM、Dense RAG、Hybrid RAG、Hybrid + Reranker 四组 80 个真实回答与 80 个完整 AI 辅助评审，另保留两条失败试评，支持检查点恢复与离线重算。没有重跑生成来删除截断或坏引用，也没有提前实现 Web UI。

消融的控制变量、指标分母、评审试运行修正、实测表格和已发现的 AI 误判见 [消融实验说明](docs/ablation_benchmark.md)。结果不能冒充独立人工指标：尤其是 100% 的可核实断言 Accuracy 和 0% 的 AI 矛盾率，不表示系统完美或真实无幻觉。

```powershell
# 无密钥即可校验已提交结果并绘图。
uv run python scripts/validate_ablation_results.py
uv run python scripts/plot_ablation_results.py
# 本地 Chunk/向量/索引/tokenizer 准备后，仅冻结输入，不调用 API。
uv run python scripts/run_ablation_benchmark.py --prepare-only
# 首次生成/评审会发送公开数据到 DeepSeek 并计费；已完成默认复用。
uv run python scripts/run_ablation_benchmark.py --allow-cloud
```

本次 AI 辅助完整性评分为 Pure 50.00%、Dense 93.33%、Hybrid 93.33%、Hybrid + Reranker 100.00%；评审已发现范围和限定词误判，最终研究结论仍需独立人工复核。

Phase 11 使用 Streamlit 原生聊天组件，提供左侧临时会话/模型选择、中间 Markdown 消息、底部聊天输入，以及回答下方的真实标题、页码、章节、URL 和 Chunk ID。默认本地 Qwen，DeepSeek 必须手动勾选云端授权；未授权不调用云端，不自动重试或回退。

```powershell
uv sync --locked
uv run streamlit run app/streamlit_app.py
# 前端交互测试使用假后端，不调用付费 API。
uv run pytest tests/test_streamlit_app.py -q
```

打开 `http://127.0.0.1:8501`。本地输入、模型与服务准备方式、授权/资源边界及验证细节见 [Streamlit 前端说明](docs/streamlit_app.md)。聊天历史只用于展示，后端每轮独立检索，不支持省略指代的多轮记忆。Phase 9 核心管线和旧实验记录保持不变；本阶段不进入 Phase 12。

## 当前目录

```text
.
├── src/apt_rag/       # Python 包
├── app/               # Streamlit 聊天展示层
├── .streamlit/        # 本机服务与页面主题配置
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
