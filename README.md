# APT-RAG：APT 情报知识问答与控制变量验证

基于检索增强生成（RAG）的研究型工程：建立可追溯 APT 报告语料和 Benchmark，比较 Embedding、向量存储、LLM 与检索/重排策略，再集成真实 Citation 和 Streamlit 聊天前端。目标是**可复现、可比较、可解释**。

**最终工程交付为 Phase 12。独立人工质量复核尚未完成。** AI 辅助评分不是最终人工结论；引用映射通过也不证明事实正确或无幻觉。

## 整体架构

```text
官方 URL + Manifest / SHA-256
 → 下载与校验 → pdfplumber 清洗 → 逐页、章节感知 Chunk / Metadata
 → 固定 BGE-M3 dense 向量 → FAISS HNSW ─┐
                            BM25 ──────┴→ 等权 RRF Top-30
 → BGE-Reranker-v2-m3 → 完整 Top-10 Context / Token 预算
 → 显式选择 DeepSeek 或本地 Qwen → 生成与响应校验
 → [S1…S10] 映射真实 Metadata → CLI / Streamlit

固定 20 题 + Ground Truth Evidence
 → Phase 4–7 组件对比 → Phase 8 冻结选型
 → Phase 9 接入验收 → Phase 10 四组消融 / AI 辅助评审 → 待独立人工复核
```

**当前 RAG 系统强制使用 LangChain（`langchain-core` / LCEL）编排**，没有非框架运行开关或自动回退。`RunnableSequence` / `RunnableBranch` 分步骤执行 Query Embedding、Dense/BM25、RRF、重排、Context、预算、LLM 与 Citation，CLI / Streamlit 继续共用 `RAGPipeline.answer()`。底层 `EmbeddingProvider`、`VectorStore`、`Retriever`、`Reranker` 和 `LLMProvider` 保持独立，沿用已验证算法及适配器。

文档解析/分块/语料向量/索引属于离线准备，复用原有脚本与固定产物，不在每次提问时重做，不替换为框架默认切块/索引/Prompt。历史 Phase 4–10 实验结果保留原实现口径，不能称为 LangChain 实测。重构范围与等价验证见 [LangChain RAG 实现](docs/langchain_rag.md)。

## Phase 0–11 交付内容

| 阶段 | 已交付内容 |
| --- | --- |
| 0 | Python 3.12、uv、包结构、依赖锁定、安全环境变量模板和测试 |
| 1 | 15 份英文 PDF 的官方 URL、下载器、大小/SHA-256；Mandiant 6 / CISA 9 |
| 2 | 491 页解析/清洗，1,297 个 Chunk，标题/厂商/物理页码/章节/URL/文本指纹 |
| 3 | `apt-qa-v1.1`：20 个中文问题，背景/TTP/攻击技术/IOC 各 5 题，真实证据和 graded relevance |
| 4 | BGE-M3、text2vec-large-chinese、m3e-base：同输入的精确余弦检索与 Recall/MRR/nDCG |
| 5 | FAISS/Chroma，同向量 HNSW，3 种规模、18 次建库、原始延迟样本、过滤/跨进程重载 |
| 6 | 本地 Qwen / DeepSeek，同 oracle context 和 Prompt，40 个真实回答及暂定辅助复核 |
| 7 | Dense / Hybrid / Hybrid + Reranker，同向量/候选预算，600 对真实神经重排分数 |
| 8 | 质量优先选型与 12 份依据指纹；保留历史选型记录的原状态字段 |
| 9 | 完整 RAG / Citation，20 题 Top-30 接入与两后端预算检查，两条真实端到端冒烟 |
| 10 | 四组 80 个真实回答、80 个 v2 AI 评审，另保留 2 个失败试评及估算费用 |
| 11 | Streamlit 聊天、临时会话、模型选择、显式云授权、loading、真实引用及前端测试 |

Phase 12 整理最终文档、复现入口、离线校验和空白人工复核工作包，不新增核心问答功能。阶段文档的“尚未进入下一阶段”属于历史边界，不代表当前状态。

## 最终技术栈

| 组件 | 固定选择 / 参数来源 |
| --- | --- |
| 环境 | Python `>=3.12,<3.13`、uv、`pyproject.toml` / `uv.lock`；原实测 Python 3.12.3 |
| 处理 | pdfplumber；逐页 Chunk 1,200 字符 / overlap 200，`processing.json` |
| Embedding | `BAAI/bge-m3` 固定 revision，CPU float32、归一化 1,024 维、256 Token，`embedding_benchmark.json` |
| 向量库 | FAISS HNSW，M=16 / efConstruction=100 / efSearch=100 / 线程 1，`vectorstore_benchmark.json` |
| 检索 | Dense + BM25、等权 RRF 常数 60、每路/融合 Top-30，`retrieval_benchmark.json` |
| 重排 | `BAAI/bge-reranker-v2-m3` 固定 revision，CPU float32、512 Token、最终 Top-10 |
| 生成 | `deepseek-flash`（云主选型）/ Ollama `qwen3.5:4b`（显式本地），temperature=0、thinking 关闭、输出 768 |
| 预算 | 共享 8,192 Token / 余量 256；Ollama 固定 `0.35.1` 与模型 digest，`rag.json` |
| 展示 | Streamlit `1.65.0`；默认本地 Qwen，无自动云回退、无多轮记忆 |
| RAG 编排 | LangChain LCEL / `langchain-core 1.6.7`；顺序节点与条件分支，跟踪上传显式关闭 |

详见 [技术栈决策](docs/stack_selection.md)。云主选型与 UI 默认本地不冲突：后者避免未经授权付费。没有实测 Milvus、其他 Reranker 或 Dense + Reranker，不补作结论。

本地问答另支持逐版核验的 Ollama `0.40.0`、`0.40.1` + 固定 `ggml` 子模型，使用独立的 `configs/ollama_runtime.json` 精确版本允许列表；历史 `0.35.1` 实验配置和结果不变。升级后出现版本/digest 报错时，见 [本地运行兼容说明](docs/ollama_runtime.md)。未知版本不会自动放行。

## 从 clone 开始安装与准备

需要 Git、uv、Python 3.12、curl 和可访问报告/模型来源的网络。实测平台为 Windows / PowerShell / CPU；未验证全部系统或 CUDA。模型/依赖需要较多磁盘和内存，未测最低资源要求。参考 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)。

```powershell
git clone https://github.com/w-harda/RAG.git
cd RAG
uv sync --locked
uv run python --version
uv run pytest -q
uv run python scripts/validate_manifest.py
uv run python scripts/validate_embedding_results.py
uv run python scripts/validate_stack_selection.py
```

这些测试/校验不需要密钥、模型或 PDF。**仅 uv sync 不会准备报告、向量、权重、tokenizer、索引和 Ollama。** 完整步骤见 [从零复现指南](docs/reproduction.md)。

### 数据与运行缓存

```powershell
uv run python scripts/download_reports.py
uv run python scripts/download_reports.py --verify-only
uv run python scripts/process_corpus.py
uv run python scripts/validate_processed.py
uv run python scripts/validate_benchmark.py
# 只重建运行所需的 BGE-M3 缓存；新结果/耗时写入 ignored 目录。
uv run python scripts/run_embedding_benchmark.py --config configs/reproduction_embedding.json --model bge-m3
uv run python scripts/validate_retrieval_results.py --verify-inputs
```

PDF、处理全文、向量、索引与权重不进入 Git。下载拒绝来源文件哈希变化。**不要直接用 Phase 4/5 默认实验命令准备 UI**：它们会覆盖历史结果；复现配置只改变结果输出目录。管线严格核验向量字节指纹，跨硬件不保证一致，不一致时停止，不篡改指纹绕过。

接着按 [复现指南](docs/reproduction.md) 下载固定 Reranker/Qwen tokenizer，执行 `prepare_rag.py --download-tokenizer`；本地问答还需对应 Ollama 和固定模型。

### 环境变量

`.env.example` 仅为模板。程序读取启动进程的 `OLLAMA_HOST`、`DEEPSEEK_API_KEY`，**不自动加载 .env**，不创建或提交真实密钥。默认 Ollama 为 `http://localhost:11434`，其他本机回环端口通过环境变量设置。本地不需要云密钥；云端需要有效密钥/余额，问题和公开片段会发送给提供方。不要输入敏感情报或真实密钥。

## 完整 RAG 与前端运行

输入、权重、tokenizer、索引和服务准备后，在仓库根目录运行：

```powershell
uv run python scripts/ask_rag.py "APT44 通常还被称为什么？它被归属于俄罗斯哪个军事情报单位，其行动范围有何特点？" --provider ollama-qwen
# 不生成、不读取云密钥，但实际编码/重排并检查预算。
uv run python scripts/ask_rag.py "APT44 的组织背景是什么？" --dry-run
# 显式云授权，调用可能计费。
uv run python scripts/ask_rag.py "APT44 的组织背景是什么？" --allow-cloud
uv run streamlit run app/streamlit_app.py
```

打开 `http://127.0.0.1:8501`。左侧临时会话/模型，中间 Markdown 消息，底部输入完整问题。DeepSeek 必须手动勾选授权，切换会话/模型撤销；失败不自动重试/回退。引用展示真实标题、物理页码、章节、URL、Chunk ID。

历史仅用于展示，存于服务会话内存；每轮独立检索，不发送历史。首次装载/CPU 重排可能超过一分钟。详见 [RAG 管线](docs/rag_pipeline.md)、[Streamlit 前端](docs/streamlit_app.md)。

## 各实验复现入口

离线核验已提交数据不调用模型/API、不重写结果：

```powershell
uv run python scripts/validate_embedding_results.py
uv run python scripts/validate_vectorstore_results.py
uv run python scripts/validate_llm_results.py
uv run python scripts/validate_retrieval_results.py
uv run python scripts/validate_stack_selection.py
uv run python scripts/validate_rag_results.py
uv run python scripts/validate_ablation_results.py
```

| 阶段 | 重新实测 / 准备入口 | 注意事项 |
| --- | --- | --- |
| 4 | `run_embedding_benchmark.py --config ...` | 三模型；新结果/缓存路径，默认会覆盖旧记录 |
| 5 | `run_vectorstore_benchmark.py --config ...` | 固定向量；新结果/索引路径，默认会覆盖旧记录 |
| 6 | `run_llm_benchmark.py --config ... --prepare-only`；去掉 prepare-only 才生成 | oracle context；首次云调用计费，已完成复用；不是生产 RAG |
| 7 | `run_retrieval_benchmark.py --config ...` | 新结果/reranker 缓存；已完成复用 |
| 8 | `validate_stack_selection.py` | 只核验历史选型，不自动调参 |
| 9 | `prepare_rag.py`、`ask_rag.py` | 接通 / 真实端到端；输出拒绝覆盖 |
| 10 | `run_ablation_benchmark.py --prepare-only` / `--allow-cloud` | 新实验 80 答 + 80 AI 评审计费；v2 独立于历史 pilot |

`plot_*_results.py` 从 JSON 绘图，默认写对应 PNG，本阶段不重绘旧图。独立配置与具体命令见 [复现指南](docs/reproduction.md)，口径见阶段文档。

## 主要实验结果

固定 15 报告 / 1,297 Chunk / 20 中文问题；小样本、单次质量实验，不能外推普遍优劣或统计显著性。[最终实验汇总](docs/results_summary.md) 提供全部主要表格、口径及原始文件链接。

- Phase 4：BGE-M3 Recall@10 **0.9000** / 全排名 MRR **0.5744**；m3e-base 为 0.6750 / 0.2864，text2vec 为 0.4167 / 0.2465。
- Phase 5：完整语料 FAISS/Chroma ANN Recall@10 均为 1.0000；无过滤 p95 约 **0.380 / 3.832 ms**、建库均值约 **0.192 / 1.887 s**。当前过滤路径不同，Chroma 过滤 p95 更低。
- Phase 6：标准证据下，本地/云端生成中位约 **3.99 / 1.44 s**，必答事实暂定辅助覆盖率 **79.58% / 95.00%**；不是人工准确率。
- Phase 7：Dense / Hybrid / Hybrid + Reranker MRR@10 为 **0.5684 / 0.4476 / 0.8458**。Hybrid 单独下降；重排 CPU 单题中位约 **31.15 s**，未证明 BM25 必需。
- Phase 9：20 固定查询 Top-30/预算通过，两后端真实端到端冒烟，不是质量 Benchmark。

Phase 10 为 **AI Judge 宏平均代理值，待独立人工复核**。Accuracy 排除无法核实断言；幻觉列仅为参考范围内 AI 矛盾判定。无引用为 N/A。

| 组 | Accuracy | 完整性 | 引用正确率 | AI 矛盾率代理 | 无法核实率 | Accuracy 有效题数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure | 81.11% | 50.00% | N/A | 5.96% | 66.98% | 15/20 |
| Dense | 96.46% | 93.33% | 98.42% | 2.71% | 39.47% | 20/20 |
| Hybrid | 93.54% | 93.33% | 91.44% | 2.17% | 41.00% | 20/20 |
| Hybrid + Reranker | 100.00% | 100.00% | 99.12% | 0.00% | 39.44% | 19/20 |

最后一组仍有约 39% 断言无法核实、一题 Accuracy 无有效分母；不能宣称完美、零幻觉或显著优越。

## 人工复核与已知限制

```powershell
# 仅导出材料，无 AI 调用/填分；已有 Chunk 时加入全文供本机复核。
uv run python scripts/prepare_human_review.py --include-text
```

导出 80 答工作包、空白逐答/必答事实/断言/引用表及协调映射，评分全部留空，隐藏策略标签但不是严格双盲。位于 ignored `data/processed/human-review-v1/`，拒绝覆盖已有目录。无 Chunk 可去掉 include-text 并依官方报告查证。详见 [人工复核流程](docs/human_review.md)。**准备材料不等于完成复核。**

- 英文 PDF、集中来源、20 题非独立大测试集、非穷尽 qrels，不代表当前全球 APT 情报。
- 双栏交错、符号缺失、跨页截断、必答项范围不一致；修订须版本化并重跑，不覆盖旧数据。
- Embedding 256 / Reranker 512 Token 有截断；生成超预算拒绝、不静默裁剪。`[S数字]` 与 ATT&CK `[S0183]` 存在引用命名冲突。
- Phase 10 同家族自评、参考审计与试评后修正存在偏差；保留截断/坏引用/失败试评，不筛好样本。
- 向量字节和 Ollama 调试模板接口严格绑定；跨设备/服务升级不保证通过。云权重不可锁定，temperature=0 不保证逐字复现；价格仅为历史估算、非账单。
- CPU 重排慢，无生产 SLA/多用户压力验证/登录/多轮记忆/Agent/MCP/插件；默认不暴露公网。
- 第三方报告/模型遵守各自版权和许可证；根目录尚无项目 LICENSE，不推定所有代码可无条件再许可。

## 项目结构与最终验收

```text
RAG/
├── src/apt_rag/           # corpus/ingestion/chunking/benchmark/embedding/vectorstore
│                         # retrieval/reranking/generation/rag/evaluation
├── app/streamlit_app.py   # 展示层
├── .streamlit/            # 本机监听与主题
├── configs/               # 冻结实验/运行配置与独立缓存复现配置
├── data/manifest/         # 官方 URL / SHA-256 / schema
├── data/benchmark/        # QA v1.1 / evidence / rubric
├── data/raw/              # ignored 原 PDF，下载生成
├── data/processed/        # ignored Chunk/向量/索引/tokenizer/人工工作包
├── results/phase4…10/     # 原始记录、摘要和图表；收尾不重写
├── scripts/               # 下载/实验/校验/绘图/复核材料导出
├── tests/                 # 自动化测试，无付费调用
├── docs/                  # 历史阶段 + 最终复现/汇总/人工复核/验收
├── .env.example           # 无真实密钥，不自动加载
├── .python-version        # 3.12
├── pyproject.toml
└── uv.lock                # 已提交的依赖锁
```

完整验收命令与真实边界见 [最终交付检查](docs/final_delivery.md)：全部测试、阶段校验、输入核验、构建和 git diff --check。已有缓存的通过不冒充全新硬件复现，人工空表不冒充最终研究结论。项目在 Phase 12 停止新增功能。
