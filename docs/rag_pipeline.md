# Phase 9：完整 RAG Pipeline 与真实来源引用

> 本文保留该阶段交付时的历史状态和实测口径；当前最终状态与完整流程见 [README](../README.md)、[复现指南](reproduction.md) 和 [最终实验汇总](results_summary.md)。后续阶段已实现的内容不回写历史实验数据。

## 已实现范围与最小架构

基于 Phase 8 选型接通：已有 PDF 加载/清洗/分块 → 固定 BGE-M3 语料向量 → FAISS HNSW → BM25/RRF → BGE Reranker → 完整 Chunk Context → 显式 LLM → 来源映射。新增的是组件编排、Context/引用、长度与身份保护，以及命令行入口；没有实现 Phase 10 消融或 Phase 11 Web UI。

核心 `RAGPipeline` 通过注入 Embedding、Retriever、Reranker、LLM 和预算计数器组合，测试可用假组件，不绑定框架。`runtime.py` 才负责读取配置、校验 Corpus、加载索引和模型。现在需要该工厂才能把已选组件接通；不创建 UI、自动调参或后续实验框架。

原 `scripts/validate_stack_selection.py` 的核心验证移到 `apt_rag.stack_selection`，脚本继续提供原 CLI 和导出函数，旧测试通过。这个小范围调整是为了让管线复用选型校验，而不是从核心代码反向调用脚本或复制校验。Phase 8 的选型 JSON 和依据指纹未改动，其 `selected_not_integrated` 表示当时状态，Phase 9 接入结果另存。

`configs/rag.json` 只新增运行路径、共享长度预算、服务版本和 tokenizer 身份，模型/Top-K/Prompt 参数仍引用 Phase 8 冻结配置。Hugging Face Hub、Tokenizers 已是现有依赖，本阶段声明为直接依赖，锁定的实际包版本未变。

## 固定输入与索引接入

沿用 15 份报告、1,297 Chunk、BGE-M3 revision、256 Token 上限、CPU/float32、归一化向量，不修改清洗、问题或标签。索引准备使用已核验的 Phase 4 Corpus/Query 缓存，而正常问答每次**重新编码用户问题、实际重排候选**，不读取标准答案来选 Context。

运行工厂检查 Chunk/文本/模型指纹、原 Query 缓存、向量字节 SHA-256、选型依据，以及索引/Metadata 文件 SHA-256。公开语料限定为该冻结 manifest/Corpus，不允许把新文件混入后仍冒充原版本。新增原报告须先版本化输入并重做有关验证。

FAISS 沿用 M=16、efConstruction=100、efSearch=100、线程 1。本阶段按原 Corpus 顺序插入，便于统一并列规则，与 Phase 5 的 seed 排列插入不同，**不作为新性能对比**。Dense 返回 Top-30；多取一个只检查候选边界同分，有边界同分时枚举本次小规模索引、按 Corpus 顺序决定，最终预算仍为 30。没有提高 efSearch 来迎合结果。

真实验证 20 个原查询：FAISS 有序 Top-30 与 NumPy 精确参考全部一致，分数误差在 2e-5 内，融合后的有序候选也与 Phase 7 完全一致，ANN Recall@30 全为 1。结果见 `results/phase9/retrieval_integration.json`。未通过就拒绝生成；失败报告写到 ignored 缓存，不覆盖 Phase 7。索引文件损坏、身份变化或未验证均明确失败。

该一致性结论只覆盖这 20 个固定查询。新用户问题仍使用 HNSW 近似检索，不保证所有问题都与全量精确搜索相同或证据召回率为 100%。

## Context、预算与引用规则

- 原问题不改写；每路 Top-30、等权 RRF 常数 60、Hybrid Top-30 重排后 Top-10。Embedding 输入超过 256 Token 时拒绝问题，不静默截断；原模型对 Corpus 的截断仍保持 Phase 4 行为。
- Top-10 的完整 Chunk 原文全部进入 Context，不按 Ground Truth 挑选，不截断正文，不偷偷减少来源。原 Prompt 保持不变。
- `S1…S10` 按本次重排顺序分配。来源表保存真实 Chunk ID、报告 ID、标题、厂商、物理 PDF 页码、section、URL 和文本指纹。URL/页码不从模型文本解析生成；章节可能为空，Chunk ID 是段落定位依据，不是另造段落号。
- 模型可引用 `[S1]` 或 `[S1, S2]`。未知、畸形、未闭合或大小写错误标签拒绝；`S<number>` 是保留来源空间，ATT&CK 软件编号若用这种方括号格式也会拒绝，不能与来源混用。
- 有 Context 却完全没有引用的回答拒绝，仅接受无引用的固定弃答句“证据不足，无法依据当前报告回答。”；检索没有任何候选则由程序返回该句，不调用生成。Dense 通常总能返回候选，**不表示其必然能回答问题**；语义缺证据仍需模型依提示弃答/限定，以及后续人工评审。
- 引用校验只证明标签和 Metadata 映射有效，不能证明每个事实句都被支持、未漏答或没有幻觉。模型生成的普通文本（包括 IOC URL）不会成为引用 Metadata。

两后端均采用 **8,192 Token 的共享工程窗口**，预留输出 768 Token、安全余量 256 Token，不因云端窗口更大而增加 Context。超限在生成前失败，不静默截断。输出空白、模型名不符、非正常 `finish_reason`（包括 `length`）、实际 usage 超预算均不接受；没有自动重试或模型回退。

### 本地 Tokenizer

通过 `hugging-face:hf-cli` 下载并校验 `Qwen/Qwen3.5-4B` 的 tokenizer 文件，固定 revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，不下载另一套生成权重。运行时逐项核对 tokenizer 有效词表、merges 与固定 GGUF 元数据；GGUF 多出的 250 个 `[PADn]` 填充项单独核验，不计作有效词表差异。

Ollama 固定 `0.35.1` 和原模型 digest。利用该版本官方源码中的 `_debug_render_only` 获取**服务实际渲染模板**，再用已匹配 tokenizer 计数；检查完整 system/user 内容都仍在模板中。生成请求带 `truncate=false`、`shift=false`，避免服务静默移除消息或移动窗口。预检会加载本地服务模型但不执行生成。

这是版本相关的调试接口，不声称是稳定跨版本 API。版本或接口/模板无法核验时明确停止，不回退字符估算。独立来源：[Ollama v0.35.1 请求类型](https://github.com/ollama/ollama/blob/v0.35.1/api/types.go)、[服务实现](https://github.com/ollama/ollama/blob/v0.35.1/server/routes.go)。

### 云端 Tokenizer

使用[官方 Token 用量页面](https://api-docs.deepseek.com/quick_start/token_usage/)提供的 V4 tokenizer ZIP，固定压缩包和两个数据文件 SHA-256，仅提取数据，不执行示例 Python/远程代码。官方模板仅用于两消息计数估算，不保证与 V4.1 云端实际模板完全相同；最终 usage 才是真实计数，超过估算加 256 余量则拒绝。

官方[模型窗口文档](https://api-docs.deepseek.com/quick_start/pricing/)支持当前服务窗口，但本项目仍用共享 8,192 预算，不把包中的 `model_max_length` 当云端窗口，也不把演示 tokenizer 当不可变云权重身份。正常生成记录返回模型/fingerprint；同一 Pipeline 会话 fingerprint 改变会停止。跨调用云模型不能完全锁定，后续实验须继续审计。

20 题完整检索 Context 的预检最大值：DeepSeek 演示 tokenizer 4,184、本地原生模板 tokenizer 4,750。均加上输出/安全预留后通过。这些 Context 来自**相同候选与模型已核验的 Phase 7 排名复用**，仅用于全部问题预算预检，没有假装重新推理 600 个重排分数。

## 数据边界与失败行为

默认选型为 DeepSeek，但 CLI 正常生成必须显式 `--allow-cloud`，表示允许将本次问题和已核验公开片段发送给提供方并计费；库工厂同样要求 `allow_cloud=True`。**问题本身也会发送**，不要在云模式输入内部敏感信息。Corpus 的公开性不能自动证明用户问题不敏感。

本地必须显式 `--provider ollama-qwen`，继续只允许回环服务、禁用代理，并核对 digest。缺少密钥、模型或缓存、身份变化/服务失败均报错，没有云↔本地自动切换。只读环境变量，不创建包含密钥的 `.env`；云端 dry-run 不读取密钥或访问云 API。

指定输出文件已经存在时，在加载模型/调用 API 前失败，防止误覆盖或不必要重复收费。验证失败的已返回响应若指定了 `--output`，以 `status=rejected` 保存原始生成和原因，不展示为通过校验的答案；网络失败请求是否收费只能查实际账单。结果不保存完整报告 Context，仅保存指纹、候选、logit、真实来源和回答。个人问答建议保存到 ignored `data/processed/`；写入 `results/` 的冒烟只允许原固定公开问题。

## 下载、准备与运行

前提：按 Phase 1/2 获取并处理原报告，按 Phase 4 重建固定 Embedding 缓存；Phase 7 Reranker 权重下载方式见其文档。无需改旧配置，不会把 PDF、权重、向量或索引推送 Git。

```powershell
uv sync --locked
uv run hf download Qwen/Qwen3.5-4B tokenizer.json tokenizer_config.json chat_template.jinja --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --max-workers 2
uv run hf cache verify Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
uv run python scripts/prepare_rag.py --download-tokenizer
# 本地服务需已运行，且 OLLAMA_HOST 与既有配置一致；不会调用云端生成。
uv run python scripts/prepare_rag.py --check-budgets
```

`hf cache verify` 校验所下载的 3 个 tokenizer 文件，其他权重等文件未下载的提示是预期情况。索引准备缺少输入时明确失败，不自行替换新模型/报告；已有索引默认核验复用，损坏/不完整目录需使用新的配置路径，不自动删除。

```powershell
# 本地完整管线；会真实编码问题、重排并生成。
uv run python scripts/ask_rag.py "APT44 通常还被称为什么？它被归属于俄罗斯哪个军事情报单位，其行动范围有何特点？" --provider ollama-qwen

# 云端完整管线，需 DEEPSEEK_API_KEY；此次调用会计费。
uv run python scripts/ask_rag.py "APT44 通常还被称为什么？它被归属于俄罗斯哪个军事情报单位，其行动范围有何特点？" --allow-cloud

# 仍实际执行 Embedding/Reranker，只准备 Context/检查预算，不生成；云端不需要密钥。
uv run python scripts/ask_rag.py "APT44 通常还被称为什么？它被归属于俄罗斯哪个军事情报单位，其行动范围有何特点？" --dry-run
```

三个入口支持 `--config`；问答另支持 `--output data/processed/rag/my-answer.json`（使用不存在的文件路径）。服务密钥和主机环境变量继承原配置；不要将密钥放到命令参数。

## 两条真实端到端冒烟

同一公开 q001，在每次调用中真实编码问题、FAISS 检索、BM25/RRF、30 对神经重排、完整 Top-10 Context、对应 LLM 生成并映射 Citation；**没有用 oracle context 或缓存生成答案**。

| 单条记录 | 本地 Ollama Qwen | DeepSeek Flash |
| --- | ---: | ---: |
| 预检输入 Token | 3,057 | 2,931（估算） |
| 服务实际输入 Token | 3,057 | 2,933 |
| 实际输出 Token | 127 | 329 |
| LLM 请求墙钟 | 9.076 s | 2.207 s |
| 管线墙钟 | 74.908 s | 55.016 s |
| 最终引用来源数 | 2 | 3 |

管线墙钟包括本次编码、检索、重排、预检/身份请求和生成，不含工厂初始化加载两个 CPU 模型/索引；不能当生产延迟或 LLM 对比结论。云记录额外保存 CLI 全程时长/时间/库版本；本地首次冒烟运行的是同一管线的早期 CLI，没有这些额外字段，保留首次真实记录，不事后补造加载时间。两条都有阶段/生成耗时和身份记录。

这只是组件接通与响应/引用契约验证，未进行独立语义评分。本地回答与云回答仍可遗漏问题要求、跑题或出现未经支持的断言；本阶段不能宣称回答准确率 100% 或零幻觉。云费用未核对账单，没有把旧 oracle 20 题费用搬成此次费用。

## 验证与阶段交付

只使用 Git 中 JSON 做离线验证，不需要模型、密钥、原始报告或服务：

```powershell
uv run python scripts/validate_rag_results.py
uv run pytest -q
uv run python scripts/validate_stack_selection.py
uv run python scripts/validate_vectorstore_results.py
uv run python scripts/validate_llm_results.py
uv run python scripts/validate_retrieval_results.py
uv build
```

已有本地 Chunk/向量可进一步复核完整 Context 与 Metadata，不加载模型：

```powershell
uv run python scripts/validate_rag_results.py --verify-inputs
```

测试覆盖完整编排/真实来源重排、未知/畸形/缺失引用、超限先于生成、弃答、未授权云端、非公开 Corpus、输出截断、坏文本指纹、共享窗口/实际 usage、FAISS 边界并列、原 Phase 6 参数不变、dry-run 禁止生成和已保存真实结果离线重算。旧阶段测试保持通过。

交付验证：71 项测试通过，Phase 5–8 回归验证通过，`uv build` 成功；另外在子进程清除 `DEEPSEEK_API_KEY` 后真实运行云端 `--dry-run`，成功完成检索、重排和预算检查，未调用云端生成。

新增：`configs/rag.json`；`src/apt_rag/rag/{__init__,context,pipeline,budget,runtime}.py`；复用的 `src/apt_rag/stack_selection.py`；三个脚本 `prepare_rag.py`、`ask_rag.py`、`validate_rag_results.py`；测试/本文档；`results/phase9/` 的接入记录和两条冒烟 JSON。

修改：README、原选型校验 CLI、Embedding Provider（可选本地缓存/拒绝截断，旧默认行为不变）、Ollama Provider（仅严格上下文模式增加两个参数）、依赖声明/锁文件。没有改变旧数据、Prompt、Benchmark 或历史实验结果。

本阶段完成管线和 Citation 接通；尚未完成独立质量评分、四组消融、稳定负载测试、数据缺陷修订和 Web UI。这些留到后续阶段，完成提交后停在 Phase 9。

建议提交说明：`feat: integrate traceable APT RAG pipeline and citations`
