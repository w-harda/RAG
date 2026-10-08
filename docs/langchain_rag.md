# 强制 LangChain RAG 编排

## 改造目标与边界

按照项目要求，当前在线 RAG 实现强制基于 LangChain，使用官方基础编排包 `langchain-core` 的 LCEL（LangChain Expression Language）。锁定版本为 `1.6.7`；`langsmith 0.14.4` 是配套依赖，直接声明只为显式关闭其跟踪上下文，不开通跟踪服务、不需要账户或密钥。

生产实现没有 native/langchain 选择开关，不会因框架失败回退到旧管线。原实现只作为 `tests/fixtures/legacy_rag_pipeline.py` 的测试快照，不进入运行时装配或发行包。

本次改变**编排方式**，不改变 Corpus、Chunk、Query、模型/revision/digest、FAISS、BM25、RRF、重排预算、Prompt、LLM 参数、Citation 或云端授权。所有旧依赖版本保留，仅增加框架及必要依赖。已有 Ollama `0.40.0` 兼容修复继续有效；经用户授权和单独接口核验后，兼容列表另明确加入 `0.40.1`，不通配新版。

## 离线准备与在线执行

完整系统仍分为两段：

1. 离线准备：Manifest → 下载 → PDF 解析/清洗 → 分块/Metadata → Corpus Embedding → FAISS 建库及验证。复用原 `process_corpus.py`、Embedding 准备和 `prepare_rag.py`，不重新切块或写回历史数据。
2. 在线问答：读取已核验的 Chunk/向量/索引并装配组件后，由 LangChain 节点链完成 Query Embedding → Dense/BM25 → RRF → Reranker → Context → 预算/身份 → LLM → Citation。

LangChain 负责在线流水线编排，不声称离线解析器和分块器改成了 LangChain 内置实现。不把文档处理和重建索引塞进每次问题请求。

## 代码与节点

入口不变：

- `scripts/ask_rag.py` → `build_pipeline()` → `pipeline.answer()`。
- `app/streamlit_app.py` → 相同工厂与 `answer()`，仍仅负责展示。

`src/apt_rag/rag/pipeline.py` 的 `RAGPipeline` 构造真实 `RunnableSequence` / `RunnableBranch`，使用 `RunnableLambda` 逐步调用组件，**不是把整个旧 `answer()` 包成一个节点**。

| LCEL 节点 | 当前函数 | 保持不变的行为 |
| --- | --- | --- |
| validate_question | `_validate_question()` | 原问题、空问题/长度拒绝，不改写 Query |
| embed_query | `_embed_query()` | 固定 Embedding 接口及参数 |
| dense_retrieval | `_dense_retrieval()` | 原 FAISS 适配器、Top-30 与并列排序 |
| bm25_retrieval | `_sparse_retrieval()` | 原词法切分、BM25、正分候选 |
| rrf_fusion | `_fuse_candidates()` | 原 `fuse_rrf()`、常数 60、等权 |
| rerank_top_k | `_rerank_candidates()` | 原 BGE Reranker，排序后 Top-10 |
| construct_context | `_build_context()` | 原 `construct_context()` 与完整 Chunk/Metadata |
| cloud_input_policy | `_check_cloud_policy()` | 未核验语料在预算/生成前拒绝发送云端 |
| token_budget | `_measure_budget()` | 原 tokenizer、模板、完整输入与输出预留 |
| model_identity | `_identify_model()` | 原 Provider 身份及 Ollama 兼容检查 |
| llm_generation | `_generate()` | 原 `generate(messages)`，不替换 HTTP 参数 |
| validate_generation_usage | `_validate_generation()` | 模型/结束原因/usage/fingerprint 校验 |
| resolve_citations | `_bind_citations()` | 原来源标签格式、存在性与缺引用检查 |
| accept_answer | `_accept_answer()` | 返回原输出契约，成功后才更新 fingerprint |

条件分支：无证据直接拒答，不调用预算或 LLM；有证据先核验云端政策和预算；`dry_run` 在身份/生成之前返回。执行顺序固定，没有 Agent、自动重试、工具规划、并发检索、查询改写或新多轮记忆。

`retrieve()`、`answer(question, dry_run=False)` 的签名、公共返回字段、错误类型和拒绝记录保持兼容。节点内部临时字段不会出现在返回值中。耗时仍按原阶段分组，但包含框架执行开销，不宣称与旧实现耗时相同。

## 安全与审计

`retrieve()` / `answer()` 使用 `tracing_context(enabled=False, parent=False)`，即使外部环境设置了 `LANGSMITH_TRACING` / `LANGCHAIN_TRACING_V2`，也不上传问题、报告或回答。不将 LCEL 图的通用 batch/async 能力包装成新公共入口，不宣称共享 mutable Pipeline 支持并发。

真实装配的 `controls.orchestration` 标识 `langchain`、包版本和跟踪关闭状态；运行环境记录框架依赖。原 `rag_config_fingerprint`、语料和索引身份不改，框架不是新增实验变量。Streamlit 缓存键纳入框架版本和 Ollama 兼容配置指纹，避免复用旧编排资源。

## 运行与验证

```powershell
uv sync --locked
$env:OLLAMA_HOST = "http://127.0.0.1:12000"
uv run python scripts/ask_rag.py "APT29 常使用哪些初始访问技术？" --provider ollama-qwen
uv run streamlit run app/streamlit_app.py
```

端口示例对应本机既有服务，其他机器按实际回环地址设置。DeepSeek 仍需安全环境变量与显式授权，本次不调用付费生成。完整数据准备方法沿用 [复现指南](reproduction.md)。

```powershell
uv run pytest tests/test_langchain_pipeline.py tests/test_rag.py tests/test_streamlit_app.py -q
uv run pytest -q
uv run python scripts/validate_rag_results.py
uv run python scripts/validate_ablation_results.py
uv build
git diff --check
```

等价测试使用同一输入和假组件，对比旧快照与新链的排名、Context/messages 指纹、Metadata、回答/引用结果、错误与拒绝记录。覆盖正常、dry-run、无证据、非法引用、超预算、输入非法、云端政策、fingerprint 和多次请求状态；检查 LCEL 的真实节点结构及顺序、环境/父上下文跟踪禁用。

额外本机核验：固定 20 查询，使用真实 FAISS/BM25/RRF、固定缓存 Query 向量和已核验重排分数，新旧输出的候选/Top-10/完整 Context/messages/来源完全一致。此验证没有重新运行神经模型或质量 Benchmark，不写回任何结果。

### 本次实际验收（2026-10-08）

- 重构首次验收的 **148 项测试**通过；随后补充 `0.40.1` 兼容拒绝/审计测试，最终 **157 项测试**通过。`uv sync --locked`、Manifest / 处理结果 / Benchmark / Phase 4–10 结果校验、`uv build`、`git diff --check` 通过。
- 49 个受检历史配置、数据、结果/图表及本地索引文件的 SHA-256 与重构前一致；原锁文件中 145 个包的版本全部保留，新增框架及其必要依赖。构建 wheel 确认强制依赖 LangChain，且不包含旧管线测试快照。
- 原 CLI 真实执行 `--provider deepseek-flash --dry-run` 成功：实际加载 BGE-M3 / Reranker，执行新 LCEL 检索链并检查预算；本条输入估算为 3,131 Token，未调用云端 API、未写实验结果。这不是完整生成或云端实际 usage 的证明。
- 首次 CLI 本地生成尝试被正确拒绝：本机 Ollama 已升级为当时未核验的 `0.40.1`。之后用户明确授权兼容核验，真实 20 题完整原生模板/预算与历史记录一致；相关接口源码核对也通过，再将该精确版本加入允许列表。后续真实生成验收见 [Ollama 运行兼容说明](ollama_runtime.md)，不能用此前 `0.40.0` 的生成记录代替。
- 新管线在 `0.40.1` 上的真实 CLI 与 Streamlit AppTest 问答均成功，输入/输出 3,348/375 Token，返回/展示 10 条 Citation；实际来源字段与 Chunk 一致。临时网页服务健康检查 HTTP 200。没有调用 DeepSeek 生成，也没有改写历史实验。这是本机运行验收，不是回答语义的人工质量结论。

历史 Phase 4–10 的数值、图表、结论口径不改，不称为 LangChain 新实验；本次等价验证也不能证明框架提升准确率、降低幻觉或改善速度。独立人工质量复核仍待完成。

## 原代码讲解材料的影响

清洗/分块、RRF 公式、BGE Reranker、Context 和 Citation 的算法说明可以保留。原 `RAGPipeline.retrieve()` 的大块编排截图需更新为 LCEL 节点构造及对应小函数；重排调用现在位于 `_rerank_candidates()`。不需要把所有算法重新称作“LangChain 算法”。本次不修改 PPT。

## 官方依据

- [RunnableSequence](https://reference.langchain.com/python/langchain-core/runnables/base/RunnableSequence)
- [RunnableLambda](https://reference.langchain.com/python/langchain-core/runnables/base/RunnableLambda)
- [RunnableBranch](https://reference.langchain.com/python/langchain-core/runnables/branch/RunnableBranch)
- [跟踪上下文控制](https://docs.langchain.com/langsmith/trace-without-env-vars)

接口以当前官方资料及锁定安装包实际签名为准，不使用旧教程的弃用链构造 API。
