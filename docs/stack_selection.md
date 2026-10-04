# Phase 8：最终研究技术栈决策

> 本文保留该阶段交付时的历史状态和实测口径；当前最终状态与完整流程见 [README](../README.md)、[复现指南](reproduction.md) 和 [最终实验汇总](results_summary.md)。后续阶段已实现的内容不回写历史实验数据。

## 状态与目标

决定采用 `apt-rag-stack-v1` 作为 Phase 9 的实现目标，状态为 `selected_not_integrated`。**组件已选定，组合尚未实现或端到端验证**；不能把组件指标相加当作最终 RAG 的质量、时延或费用。

这是一套适用于当前公开 APT 语料的、质量优先的单机研究配置，不承诺交互延迟 SLA。选择依据为已提交的 Phase 4–7 真实记录；本阶段不重测、不调用模型、不收费、不修改 Corpus/Benchmark。决策是在看到实验后作出的工程取舍，不冒充事前注册的加权排名规则。

## 选型总表

| 组件 | 主方案 | 当前证据与取舍 |
| --- | --- | --- |
| 文档处理 | 保持 pdfplumber 和现有逐页清洗/分块 | 沿用 15 报告、1,297 Chunk，保留 Metadata 和指纹；不事后调整数据来提高分数 |
| Embedding | BGE-M3 dense，固定 revision | Phase 4 的全排名 MRR=0.5744、nDCG@10=0.6461，按原先规则胜出 |
| Vector Store | FAISS HNSW，应用管理 Metadata | 当前完整语料两库质量相同；FAISS 无过滤 p95≈0.380 ms、建库均值≈0.192 s，低于 Chroma 的≈3.832 ms/1.887 s |
| 检索 | BM25 + Dense，等权 RRF，随后重排 | Hybrid 单独不优于 Dense；采用已实测的完整重排组合，不声称 BM25 必需 |
| Reranker | BGE-Reranker-v2-m3，固定 revision | Phase 7 的 MRR@10=0.8458、nDCG@10=0.8755；接受当前 CPU 单题约 31 秒的代价 |
| 主 LLM | DeepSeek `deepseek-flash`，非思考模式 | Phase 6 oracle context 的必答事实辅助覆盖率 95%，单题中位≈1.44 s，优于当前本地方案；不是独立人工准确率 |
| 显式本地 LLM | Ollama `qwen3.5:4b`，固定 digest | 本地覆盖率≈79.58%、中位≈3.99 s；用于不允许将输入发到云端的场景，不自动切换 |
| 框架 | 复用现有框架无关接口和适配器 | 目前无需 LangChain/LlamaIndex 才能串联组件；不引入未验证的新框架/依赖 |

框架是工具而不是实验变量。用户原要求优先使用框架；在当前核心接口和适配器已完成的前提下，额外包裹框架没有本阶段必需收益。若以后需要框架组件，仅在适配层引入，不重写已验证实现。

### 为什么不选择其他候选

- **m3e-base / text2vec-large-chinese**：Phase 4 的 Recall@10 分别为 0.6750/0.4167，BGE-M3 为 0.9000。m3e 编码更快，但当前研究目标优先证据排名质量；未证明扩大预算或改任务后仍有相同排序。
- **Chroma**：原生 Metadata/过滤管理更省应用逻辑，完整语料过滤 p95≈7.367 ms，比本项目 FAISS 适配器的≈9.090 ms 低。当前目标是单机研究、主要无过滤查询，FAISS 已有可重载适配器，所以选 FAISS；不是声称 FAISS 全面更好。若后续需要高频多字段过滤，应重新评估 Chroma。Milvus 未实测，不作为本次选型依据。
- **Dense / Hybrid 不重排**：Phase 7 两者 Recall@10 都是 0.9000，MRR@10 分别为 0.5684/0.4476。重排组 Recall@10=1.0000、MRR@10=0.8458。选择完整重排组合是质量与计算开销的取舍；若必须低延迟，Dense 是已测备选，但本次不新增快速配置或承诺时延。
- **Dense + Reranker**：尚未测量，不能据此推断优劣。Dense Top-30 已覆盖全部已标注证据，BM25 没有提高候选召回上界。继续保留 Hybrid 是为了使用已测完整组合，并保留后续消融的相同候选构造；它不是 BM25 因果收益的证明。
- **本地 Qwen 作为主生成方案**：其隐私与可固定 digest 的优势明确，但当前 oracle 输入下的辅助质量较低，且存在较大冷启动。云端选择限定为公开语料研究，不表示适合内部敏感情报。

## 配置与追溯

`configs/final_stack.json` 是**选型记录而非可运行的 Pipeline 配置**：选择组件 ID，引用既有配置与摘要，并冻结 12 份 JSON 的内容指纹。没有复制模型参数到多个配置文件，避免一处改 revision、另一处未更新。

内容指纹采用与既有实验相同的 `json.dumps(ensure_ascii=False, sort_keys=True)` 后计算 SHA-256，不受 CRLF、缩进或键顺序变化影响；字段/值改变会失败。依赖仍由 `uv.lock` 约束，本阶段不新增或升级依赖。历史原始记录由各阶段验证脚本核验，选型校验不代替它们，不重新运行神经网络。

| 参数 | 固定来源及值 |
| --- | --- |
| 分块 | `processing.json`，1,200 字符，重叠 200，原逐页策略 |
| Embedding | `embedding_benchmark.json` 中 `bge-m3`；revision `5617a9f61b028005a4858fdac845db406aefb181`，1,024 维归一化 float32、CPU、256 Token 上限、空查询前缀 |
| Vector Store | `vectorstore_benchmark.json`，FAISS HNSW 内积，M=16、efConstruction=100、efSearch=100、线程 1 |
| 检索与重排 | `retrieval_benchmark.json`，原问题不改写，每路 Top-30、RRF 常数 60、等权、BM25 k1=1.2/b=0.75、正分候选；融合 Top-30→重排 Top-10 |
| Reranker | 同上，revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`，CPU/float32、batch 4、512 Token、only_second、线程 4、seed 42 |
| 生成 | `llm_benchmark.json` 选定 provider，复用温度 0、非流式、关闭 thinking、输出上限 768 Token；本地 num_ctx=8192/seed=42 |
| Prompt | `llm_benchmark_prompt.json`；报告作为证据而非指令，事实引用 `[S1]` 等真实来源标签，证据不足须明确指出 |

主生成模型不提供不可变权重 revision。复用 Phase 6 的模型名称及身份检查，运行时仍需记录返回模型、时间和 fingerprint；新 fingerprint 不能被当作旧实验的同一模型。价格只引用 Phase 6 的历史快照，本阶段不查询账户、余额或账单，也不把历史 $0.008502 的估算外推为最终系统成本。

## 架构决策：为何现在决定，哪些留到以后

1. **保持分块和 Metadata 不变**。现在需要固定后续实现的输入，解决跨阶段换语料或改标注后混比的问题。已知双栏交错/路径符号/q019 证据不全问题记录在 Phase 6 文档；修订应另开数据版本，不在本阶段悄悄修复。
2. **选择 FAISS 而非继续用 NumPy 作为最终存储**。Phase 5 已提供建库、持久化、重载及 Metadata 管理证据，现在明确接入目标避免 Phase 9 临时选后端。Phase 7 使用 NumPy 精确余弦，Phase 5 仅验证 Top-10，而 Hybrid 请求 Top-30；**FAISS Top-30 与精确结果的一致性尚未验证**，不能把 Phase 7 的指标直接归给 FAISS 组合。
3. **保留研究质量优先的重排**。现在需要决定候选预算、重排版本及代价，解决后续管线任意调整的问题。GPU、减少候选或换小模型均改变配置，留到独立实验；不擅自更换 CPU Torch。
4. **主云端、本地显式选择，不自动回退**。现在确定数据边界，防止云端失败时静默换模型破坏可比较性，也防止本地模式将输入发送到云端。主配置只处理已有公开报告；缺少密钥/身份不符必须明确失败。政策的实际强制执行属于 Phase 9，不声称本阶段已实现。
5. **先冻结引用与 Context 政策，不写 Citation/Pipeline**。Phase 9 从检索到的 Chunk Metadata 构造 Context 和标签表，不使用 Ground Truth evidence 挑选答案；LLM 不得自行生成 URL/页码。未知标签应拒绝；来源标签存在不等于断言获得支持。具体渲染与响应结构留到 Phase 9。

这里把最终 Context 的检索上限定为 Top-10，以沿用已测输出预算；**不保证 10 个完整 Chunk 加 Prompt 能放入每个生成模型窗口**。Phase 9 必须按真实 tokenizer/服务预算验证，保留输出空间。超限明确失败，不静默截取全文、丢证据或改变 Top-K。如需截断/减少片段，要版本化规则，并在所有生成/消融对照中固定同一政策。本阶段没有实现 Context 构造器。

## 证据强度与后续验收边界

- 三个候选池召回均为 100% 指固定 qrels，并非穷尽相关知识；仅 20 中文问题、主要英文报告，不支持统计显著性或泛化结论。
- Phase 4 MRR 使用完整排名，Phase 5/7 是 MRR@10；选型不能直接混算。Phase 6 使用 oracle context、单一非盲 AI 辅助评审，不能推断真实检索 Context 下的回答质量。
- Phase 7 重排中位约 31 秒，与 Phase 6 生成中位不是同一次端到端测量，不能简单相加来宣称最终延迟。云成本、并发和 GPU Reranker 均未测量。
- 当前 20 题既用于选型又将用于原要求消融，结果可能存在选择偏差；最终研究需要独立测试集/相关性标注及独立人工复核，不能把同一集合称为无偏泛化评估。

Phase 9 接入前/后必须验证：FAISS Top-30 与精确参考/候选池的一致性及偏差、固定并列规则、完整 Token 预算、真实 Metadata 标签映射、未知引用拒绝、缺证据处理、显式后端和云端数据边界。若排名变化，另存结果并重测检索，不能覆盖 Phase 7。Phase 10 再用完全相同问题、模型/Prompt/预算做四组端到端消融。

以上是后续阶段验收要求，不代表已提前实现。Benchmark、生成服务、Web UI 和最终 RAG 均未在本阶段改写。

## 独立验证与文件清单

无需 PDF、模型、向量、Ollama、密钥或网络：

```powershell
uv sync --locked
uv run python scripts/validate_stack_selection.py
uv run python scripts/validate_vectorstore_results.py
uv run python scripts/validate_llm_results.py
uv run python scripts/validate_retrieval_results.py
uv run pytest -q
```

选型脚本仅校验引用、内容指纹、组件参与情况和跨阶段输入一致性，不自动选赢家、不执行部署政策或生成回答，支持 `--config`。依据改变时必须审查和版本化选型，不自动更新 hash 来掩盖改变。小范围校验代码仅放在 `scripts/`，不为本阶段增加新核心抽象。

新增：`configs/final_stack.json`、`scripts/validate_stack_selection.py`、`tests/test_stack_selection.py`、本文档。修改：`README.md`。旧配置、实验结果、模型依赖和 `.env.example` 未改动。

本阶段完成组件决策、解释和离线验证。尚未完成组件组合实测、Top-30 后端接入验证、Context/Citation 实现、数据修订、独立人工评分、重复负载及 Web UI；按阶段留待后续，停在 Phase 8。

建议提交说明：`docs: select evidence-backed APT RAG research stack`

## 官方接口复核

本阶段查阅官方文档确认候选接口/能力，不以官方宣传替代本项目对比结果：

- [BGE-M3 模型卡](https://huggingface.co/BAAI/bge-m3)：支持 dense/Sentence Transformers 与多语言；本项目只测 dense，未使用其 sparse/ColBERT。
- [BGE Reranker 模型卡](https://huggingface.co/BAAI/bge-reranker-v2-m3)：交叉编码器输出相关性分数，沿用 Phase 7 适配器。
- [FAISS 索引文档](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)：HNSWFlat 索引类型；应用 Metadata 不属于 FAISS 原生数据库能力。
- [DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/) 和 [Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/)：当前文档列出 `deepseek-flash`、显式 `thinking.type=disabled`；文档可用不等于实际账户权限，Phase 8 未调用账户 API。

历史库/服务版本与模型身份见各阶段原始环境记录，不在本阶段按最新版本重新安装。
