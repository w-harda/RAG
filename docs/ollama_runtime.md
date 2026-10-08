# Ollama 0.40.0 / 0.40.1 本地问答兼容说明

> 后续 LangChain 重构仅改变问答编排，继续复用本文的 Ollama 兼容适配和原 `TokenBudget`。下文“`RAGPipeline.answer()` 不变”指本次兼容修复时的边界；其当前内部实现见 [LangChain RAG 实现](langchain_rag.md)。

## 问题与范围

历史实验和 Phase 9 配置固定 Ollama `0.35.1`。本机升级为 `0.40.0` 后，严格预算检查拒绝未验证版本；首次装载旧 Qwen 模型还会触发 Ollama 的后台格式迁移，使模型标签出现多个 runner/digest。

本次只修复 CLI / Streamlit 的真实本地运行兼容性。不重新生成实验结果，不改 `configs/rag.json`、Phase 6–10 配置、历史结果、索引或模型权重，不自动升级/降级 Ollama，不调用 DeepSeek。

## 最小设计

- `configs/ollama_runtime.json` 通过 `verified_ollama_versions` 精确允许已逐版核验的 `0.40.0`、`0.40.1`，另锁定历史基线、迁移后 `ggml` 子模型 digest 和模板指纹。
- `rag/runtime.py` 在真实本地管线装配时读取服务版本。`0.35.1` 沿用原配置；允许列表中的版本必须匹配兼容配置的历史基线，才建立独立的内存运行参数，并在 Provider、预算和审计中记录实际版本。空/非法/重复列表、通配符和未知版本拒绝，不自动接受“更高版本”。
- `OllamaProvider` 在身份检查、原生模板预检和生成请求中统一显式选择 `runner=ggml`，避免相同标签下默认选择 `llamacpp`。仍验证 digest、thinking 能力、runner 和模板指纹。
- 原 `TokenBudget` 和 `RAGPipeline.answer()` 不变。真实模板完整性、固定 tokenizer/vocab、输入/输出长度、禁止截断/窗口移动、实际 usage 和 Citation 校验继续执行。
- 原索引/实验配置指纹不变；运行输出的 `controls.ollama_runtime_compatibility` 记录兼容配置指纹、实际版本/runner/digest 与历史基线，生成身份也记录实际版本。不能把兼容运行记录冒充 `0.35.1` 的实验。
- Streamlit 的管线缓存键加入兼容配置文件指纹，文件变化后不复用旧管线；同一配置下仍复用模型资源。

兼容配置只适用于文件中指定的 digest。若下载了不同模型、服务继续升级或模板变化，仍会拒绝；不能随手修改指纹绕过检查。

为什么现在实现：已有网页不能回答，必须修复装配层与 Provider 的版本相关接口。为什么不用直接改 `rag.json`：该文件参与历史结果和索引身份验证，直接修改会造成历史记录不一致。为何不通配所有新版：模板预检是调试接口，必须逐版验证。

## 使用

现有索引和模型缓存不需要重建。先在运行 Streamlit 的终端按 `Ctrl+C` 停止网页服务，再执行：

```powershell
Set-Location E:\Git\RAG
$env:OLLAMA_HOST = "http://127.0.0.1:12000"
uv run streamlit run app/streamlit_app.py
```

打开 `http://127.0.0.1:8501`，刷新页面，选择本地 Qwen，重新提交问题。端口示例对应本次本机服务；其他机器应使用其实际回环 endpoint。服务重启会清空临时聊天记录。

CLI 共用同一装配层：

```powershell
uv run python scripts/ask_rag.py "APT29 常使用哪些初始访问技术？" --provider ollama-qwen
```

原 `prepare_rag.py --check-budgets` 保持历史基线验收行为，不用于在兼容新版上覆盖历史预算记录。已有索引无需为本次修复重建；在新版服务上做新的科研实验需要另建配置和输出位置，并保留实际运行身份。

## 首次 0.40.0 修复的验证记录

本次在实际 `0.40.0` 服务上验证了固定 20 题的完整检索 Context 原生模板预检：输入 Token 数、渲染模板 SHA-256、tokenizer SHA-256 与历史记录全部一致，最大输入为 4,750 Token。这里只复用固定检索排名核验接口，不重新运行检索 Benchmark，也不写回结果文件。

上述一致性不能证明两个 Ollama 版本产生完全相同的答案，也不构成回答质量提升的实验结论。模型标签迁移由 Ollama 自身执行，源格式和转换格式同时存在；本项目不删除任何模型副本。

自动化回归入口：

```powershell
uv run pytest tests/test_ollama_runtime.py tests/test_rag.py tests/test_streamlit_app.py -q
uv run pytest -q
uv run python scripts/validate_rag_results.py
uv run python scripts/validate_ablation_results.py
uv build
git diff --check
```

测试使用假 HTTP，覆盖原版回归、已验证版本、未知版本拒绝、兼容配置基线、重复模型标签、runner/digest/模板变化、原生 tokenizer/完整模板失败，以及预检和生成使用相同的严格请求参数；不调用付费 API。

真实冒烟使用与截图相同的 APT29 问题：CLI 完成回答并输出引用；Streamlit 官方 AppTest 使用真实本地后端完成两条聊天消息和 10 个 Citation 展示块，无页面错误。该次网页生成输入/输出为 3,348/375 Token，生成约 9.1 秒，管线约 54.2 秒（不含首次装载）。这是运行兼容性验证，不是人工质量复核或新增性能 Benchmark，回答内容仍需核实。

本次最终验收：`uv sync --locked`、全部 129 项测试、Manifest / 处理结果 / Benchmark / Phase 4–10 离线校验、`uv build`、`git diff --check` 通过；历史配置与 `results/` 无改动。

## 0.40.1 与 LangChain 真实问答验收（2026-10-08）

本机服务自动升级为 `0.40.1` 后，首次问答被原版本保护拒绝。用户明确授权核验后，才扩展运行配置的精确版本列表；没有降级/升级服务、切换 runner、改权重或放宽预算。

- 核对官方 `v0.40.1` 与 `v0.40.0` 的请求类型和服务源码。前者 `api/types.go` 内容 SHA-256 为 `75369e029dcfc815013243b1e728fcff32514a3b38b26f3288eb1ee2b7fa4d1b`，`server/routes.go` 为 `3fc5a208dabafa715d4ca3114aa84c8181d7d5f215c151b79224885883b060b6`；相关本地请求路径未变。
- 实际服务身份核验：`0.40.1`、`qwen3.5:4b`、`runner=ggml`，digest 和模板指纹与兼容配置一致。明确避开同标签下另一个 `llamacpp` 子模型。
- 复用固定 20 题及 Phase 7 保存的 Top-10，通过真实 `0.40.1` 原生 render-only 接口，对比 Phase 9 保存的完整 messages 指纹和全部预算字段，全部一致；原生 vocab/merges 检查通过，最大输入 4,750 Token。未重跑质量 Benchmark、未覆盖结果。
- 原 CLI 经真实 LangChain 工厂及 `answer()` 回答“APT29 常使用哪些初始访问技术？”：`status=answered`、`finish_reason=stop`，输入/输出 3,348/375 Token，10 个 Citation 可由真实 Chunk Metadata 和标签映射重算。该条生成约 10.2 秒，属于冒烟运行记录，不作为性能或质量比较结论。
- Streamlit 官方 AppTest 未替换后端，实际提交同一问题：两条聊天消息、10 个 Citation 展示块、标题/页码/章节/URL/Chunk ID 与实际 Chunk 一致，无页面错误；重新运行页面未重复提交。该条生成约 7.5 秒，管线约 44.4 秒（不含首次装载）。临时启动 `127.0.0.1:8502` 服务的健康检查 HTTP 200，验收后关闭；不停止原 `8501` 或 Ollama。
- 回归测试覆盖两个允许版本与实际版本审计，以及未知版本、空/非法/重复/通配列表拒绝。全套 **157 项测试**、`uv sync --locked`、全部阶段结果校验、`uv build`、`git diff --check` 通过；49 个受检历史配置/数据/结果/索引文件保持原 SHA-256。

CLI 单条审计记录保存在 Git 忽略的 `data/processed/runtime-verification/langchain-ollama-0401-cli.json`；不进入 `results/`，不冒充历史 `0.35.1` 实验。Streamlit 验收采用官方 AppTest，而非人工浏览器点击或视觉验收。本文仅证明接口、运行、预算、引用格式/存在性和来源映射通过；**不能证明回答中的每条事实及引用语义都正确，也不是独立人工复核。**

## 官方接口依据

- [Ollama v0.40.0 请求类型](https://github.com/ollama/ollama/blob/v0.40.0/api/types.go)：`runner`、`truncate`、`shift` 和调试模板字段。
- [Ollama v0.40.0 服务实现](https://github.com/ollama/ollama/blob/v0.40.0/server/routes.go)：原生模板预检返回及 runner 选择。
- [Ollama v0.40.0 自动兼容迁移](https://github.com/ollama/ollama/blob/v0.40.0/compatmigrate/migrate.go)：旧格式在后台迁移、保留源子模型的行为。
- [Ollama v0.40.1 请求类型](https://github.com/ollama/ollama/blob/v0.40.1/api/types.go)：与 `0.40.0` 文件逐字一致，保留固定 runner、禁止截断/窗口移动和 render-only 字段。
- [Ollama v0.40.1 服务实现](https://github.com/ollama/ollama/blob/v0.40.1/server/routes.go)：与 `0.40.0` 的差异为新增云账户 balance/usage 路由；本项目调用的身份、模型、模板预检和本地 Chat 路径未变。源码核对不代替真实服务验收。
