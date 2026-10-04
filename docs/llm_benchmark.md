# Phase 6：固定证据的 LLM 对比

> 本文保留该阶段交付时的历史状态和实测口径；当前最终状态与完整流程见 [README](../README.md)、[复现指南](reproduction.md) 和 [最终实验汇总](results_summary.md)。后续阶段已实现的内容不回写历史实验数据。

本阶段比较本地 Ollama `qwen3.5:4b` 与云端 DeepSeek `deepseek-flash`，已真实运行同一 20 题，保存 40 个回答。没有实现检索策略对比、最终 RAG Pipeline、Reranker 或 Web UI。

## 核心问题与最小方案

LLM 对比需要排除检索差异。这里采用 **标准证据 Context（oracle context）**：按 Phase 3 的 evidence 顺序取完整 Chunk，为每题冻结 system/user messages，两个后端使用完全相同的 messages。**不把 expected_answer 或必答事实清单发送给模型**。这衡量的是有标准证据时的生成表现，不是端到端 RAG 效果；不同问题的证据数量不同，但同一问题在两个模型之间完全相同。

新增框架无关的 `LLMProvider` 及统一返回记录，是因为本阶段已经需要替换两种生成后端；协议只包含身份检查和生成。直接使用 `httpx` 调用经官方文档和本机验证的 API，避免为两种 HTTP 调用引入额外框架依赖。复用现有 Benchmark/Chunk 校验、指纹和结果写入工具，不改动既有解析、Embedding 和向量存储实现。检索编排与最终 Citation 渲染放到后续阶段。

## 控制变量与模型身份

- Benchmark：`apt-qa-v1.1`，20 题；语料与问题指纹延续 Phase 4/5。
- Context：原始 evidence 对应的完整 Chunk，带真实报告标题、厂商、物理页码、URL 和 Chunk ID；没有重新分块或截取。
- Prompt：`configs/llm_benchmark_prompt.json`，所有问题采用相同模板。
- messages、输入集合、Prompt、语料及配置均保存 SHA-256 指纹。
- `temperature=0`，非流式，关闭 thinking，输出上限 768 Token，单题单次，无重试、无答案挑选。
- Ollama：`num_ctx=8192`、`seed=42`、`keep_alive=10m`，显式中和模型文件自带的 presence penalty / top-p / top-k 等采样设置。实际最大输入为 917 Token，远低于窗口上限。
- 两个后端 Tokenizer 和内建聊天模板不同，Token 数及上限不能视为完全相同的字符预算；DeepSeek 不提供与本地 seed 等价的控制。温度为零也不保证跨硬件或云端升级后逐字一致。

本地使用已安装的 GGUF，`Q4_K_M`，digest 为 `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`，Ollama `0.35.1`。API details 报告 parameter_size 为 `4.7B`，不要把 tag `4b` 当作精确参数量。身份记录还包括聊天模板指纹、模型文件大小。

DeepSeek 当前官方名称是 `deepseek-flash`，文档版本为 `DeepSeek-V4.1-Flash`；已通过 `/models` 检查可用。它不提供不可变权重 revision，因此只能记录请求/返回模型名、文档版本、时间及逐题 system_fingerprint；本次所有响应的 fingerprint 均为 `aeb56401ca74e127821c4f9126dcb669`。这不是对未来云模型版本的锁定。

两模型规格、量化和运行后端不同是这组部署方案的组成部分。结果不能解释为“相同模型只改运行位置”的因果实验。

## 真实性能结果

本机为 i7-10875H、约 16 GiB 内存、RTX 2060 Max-Q 6 GiB，实验中 `ollama ps` 显示 100% GPU。环境和身份记录位于 `results/phase6/`。

| 指标（20 题单遍） | Ollama Qwen | DeepSeek Flash |
| --- | ---: | ---: |
| 总墙钟耗时 | 176.63 s | 30.89 s |
| 单题中位耗时 | 3.99 s | 1.44 s |
| 单题 p95 耗时 | 14.93 s | 2.52 s |
| 第一个请求 | 89.82 s | 1.13 s |
| 输入 Token 合计 | 11,521 | 10,831 |
| 输出 Token 合计 | 4,084 | 4,377 |
| 输出截断数 | 0 | 0 |
| 带来源标签的回答比例 | 100% | 100% |
| 未知来源标签数 | 0 | 0 |
| API 费用估算 | 不适用 | $0.008502 |

墙钟耗时包括请求、排队、加载、预填充和生成，**不是纯解码速度，也不是首 Token 延迟**。本地首题包含较大的冷启动开销，其中 API 返回 load_duration 为 49.83 秒；本表不事后删去首题。后续 keep_alive 和服务缓存可能影响耗时。只有单遍实验，不能据此声称稳定生产 p95 或统计显著性。

云费用使用 2026-10-04 查阅的峰值美元单价：缓存命中输入 $0.006/M、未命中 $0.3/M、输出 $1.2/M，按实际 usage 计算。本次缓存命中为 0，估算是 `(10831×0.3 + 4377×1.2)/1e6`；非高峰单价可能更低。**这是保守估算，不是实际账单**，不记录账户余额或密钥。本地没有这类 API 费用，但耗电、设备采购和折旧未测量，故成本为 `null` 而非零。[DeepSeek 官方价格](https://api-docs.deepseek.com/quick_start/pricing/)

## 回答质量：辅助复核，不是独立人工评审

在调用前建立 `data/benchmark/llm_rubric_v1.json`，将标准答案分成每题 3–4 个必答复合事实。调用后由 Codex 按实际冻结 Context、答案和清单逐题辅助复核，记录在 `review.json`。每条记录绑定 answer_sha256，包含事实覆盖布尔值、每次引用的支持判定、无证据断言及具体依据。

- `required_fact_coverage`：每题完整覆盖的必答项 / 该题必答项数，再对 20 题宏平均。复合项只有全部覆盖才计 1，部分覆盖不计；可接受同义表述，观察/置信度限定需保留。
- `fully_covered`：所有必答项均覆盖的回答比例。
- `semantic_citation_support`：每题能支持相邻断言的引用标签次数 / 引用次数，再宏平均；组合引用的每个标签分别评估。无引用时记 0。
- `has_unsupported_claim`：包含至少一个无证据或与证据冲突的断言的回答比例，不是逐原子断言的幻觉率。漏答本身不记作编造。

| 初步辅助评分 | Ollama Qwen | DeepSeek Flash |
| --- | ---: | ---: |
| 必答事实覆盖率 | 79.58% | 95.00% |
| 全部必答事实覆盖的回答 | 50.00% | 85.00% |
| 引文支持率（宏平均） | 85.87% | 99.29% |
| 含无证据/冲突断言的回答 | 30.00% | 0.00% |

这些是 **单一、未盲评的 AI 辅助复核结果**，不能宣称独立人工 Accuracy、最终 Citation Correctness 或可靠的真实幻觉率。`0%` 只表示本次复核未发现，不表示模型不会产生幻觉。正式 Phase 10 仍需要独立评审、原子事实口径、评审一致性与更充分的重复实验。

自动 `citation_check` 只检查来源标签是否存在，不能证明内容正确。解析支持 `[S1, S2]`；`[S0183]` 是报告内的 ATT&CK 软件编号，不作为实验来源标签。语义支持率来自逐题辅助复核，而不是这个正则检查。

代表性差异：

- q016：Qwen 的 MD5 正确，但遗漏 32/64 位，并把交错文本中的 TeamViewer 引入 BLACKCOFFEE 指令托管解释；DeepSeek 保留了准确载荷链。
- q020：Qwen 漏掉 Pastebin URL，将第四项域名放进前三项；DeepSeek 按原文起始顺序保留三个指标。
- q007：DeepSeek 答对低频喷洒及多类 IP，但遗漏标准答案中的 MFA 豁免信息，不能只凭流畅度给满分。

## 发现的数据限制

保持 Corpus/Benchmark 不变，不在看过结果后修改标准答案以改变分数：

1. q009、q014、q016、q017 的 PDF 双栏文本存在交错，可能导致错误关联；oracle context 不等于完美清洗文本。
2. q018 的 S1 里路径为 `%windows\registration\`，缺少闭合 `%`，S2 则给出 `%windows%\Registration`。Qwen 照抄前者；辅助复核将它计为必答路径不完整，但不直接归为模型凭空编造。Qwen 的文件名结构引用 S1，实际结构只出现在 S2。
3. q019 标准答案包含伪造勒索信息，但本次 evidence Chunk 在勒索文本开头截断，未充分支持该复合必答项；两个模型都未覆盖。覆盖率受证据完备性限制，不能将该漏答全归于模型。
4. 部分标准答案包含问题没有直接要求的背景事实，例如 q007 的 MFA 豁免、q011 的起始日期；本次按预先清单严格计分，后续需要审查问题/必答项的一致性。

数据修订应单独版本化并重跑受影响实验，不能覆盖本次记录或与旧结果混比。

## 成本、隐私与部署复杂度

| 方面 | 本地 Ollama | 云端 DeepSeek |
| --- | --- | --- |
| 部署 | 需安装服务、约 3.39 GB 模型、足够内存/显存；校验 digest | 需联网、有效密钥、模型权限/余额；没有本地模型安装 |
| 实验数据 | 本阶段推理只允许本机回环 endpoint，禁用代理 | 问题和所选公开报告片段发送给提供方 |
| 本次可观测成本 | 仅记录加载与推理时间，硬件/能源成本未测 | 记录 usage 和价格快照下的估算，不称实际扣费 |
| 可复现边界 | 固定模型 digest/模板，仍受硬件和服务版本影响 | 固定输入，但不能锁定云端权重；保留时间和 fingerprint |

这不是提供方日志保留政策的审计；不得据此把云端部署视为适合敏感内部情报。

## 运行与独立验证

```powershell
uv sync --locked
uv run pytest -q
# 无需模型、密钥、处理后语料；从 Git 中记录离线重算。
uv run python scripts/validate_llm_results.py
uv run python scripts/plot_llm_results.py
```

在已具备 Phase 2 Chunk 时：

```powershell
# 不调用 API，不需要密钥，仅验证和冻结完整输入。
uv run python scripts/run_llm_benchmark.py --prepare-only
uv run python scripts/validate_llm_results.py --verify-inputs

# 首次运行新实验会调用模型并产生 API 费用；已完成记录只复用。
uv run python scripts/run_llm_benchmark.py
uv run python scripts/run_llm_benchmark.py --provider ollama
uv run python scripts/run_llm_benchmark.py --provider deepseek
```

程序只读取进程环境中的 `OLLAMA_HOST` 和 `DEEPSEEK_API_KEY`，不自动加载 `.env`。`.env.example` 只提供空密钥字段。不要将密钥贴入命令记录、文档或 Git；在本机安全的环境变量配置界面设置后重启终端/应用以继承。不会读取 `OPENAI_API_KEY`。

本地身份不同会直接报错，不能静默拉取/替换另一个模型；云端返回的 backend fingerprint 变化时也停止旧实验。DeepSeek 只接受官方 HTTPS endpoint。HTTP 错误只输出状态码及安全摘要，不保存服务端错误正文或请求头。失败时停止，已有逐题检查点保留并在续跑前校验；没有自动重试或更好的答案筛选。费用统计仅包括保存在结果中的成功回答，失败/中断请求是否计费需另查实际账单。

**本次记录已完成，直接运行不会重测。** 重新实测：复制配置为新文件，修改 `results_directory`（以及需要变更的 provider ID/身份），用 `--config` 指定它；可以沿用同一冻结输入。不要删除/覆盖已提交的原始结果。新实验的复核必须重新绑定新答案，不能复制旧评分。运行、离线验证和绘图入口均支持 `--config`；单后端运行后需运行 `--provider all` 汇总，两侧已完成的回答都会复用。

## 文件清单与当前尚未完成事项

新增：两个配置文件、必答事实清单、`generation` 适配器与协议、`evaluation/llm_benchmark.py`、`evaluation/llm_review.py`、三个运行/校验/绘图脚本、测试文件、本阶段输入审计/原始回答/逐题复核/摘要/环境记录/图表。

修改：`pyproject.toml` 与 `uv.lock` 将已使用的 httpx 声明为直接依赖，`.env.example` 添加安全变量模板，README 更新进度和命令。没有改动旧阶段数据和结果。

本阶段已完成实现与真实初步对比；独立人工盲评、已列数据缺陷的版本化修订、稳定负载/重复性能实验和本地真实运行成本核算尚未完成。尚未推进 Phase 7 及以后。

提交说明：`feat: benchmark local Qwen and DeepSeek with frozen APT evidence`

## 官方 API 依据

- [Ollama Chat API](https://docs.ollama.com/api/chat)：使用 `/api/chat`、非流式、显式关闭 think，并保存用量和时长字段。
- [Ollama 模型列表](https://docs.ollama.com/api/tags)：读取 digest；详情/版本在本机 `/api/show`、`/api/version` 验证。
- [Ollama 参数文档](https://docs.ollama.com/modelfile)：Context、采样、seed 和输出上限的参数依据。
- [DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/) 与 [Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/)：当前模型名和显式关闭思考的请求格式；未使用旧教程假设的弃用模型名。
