# 从 clone 到 RAG / Streamlit 的复现指南

本指南区分三种行为：**离线校验历史记录**、**重建运行缓存并实际问答**、**独立重新实测**。三者不能互相冒充；历史数据、固定配置、Prompt 和模型身份均不在收尾时修改。

命令以 Windows PowerShell 为例，在仓库根目录执行；原实验为 Python 3.12.3 / Windows / CPU。已提交依赖锁不等于锁定所有硬件、操作系统、云权重或外部下载服务。请逐条检查退出码，失败时不要继续后面的命令。官方来源文件改变、向量字节不同或模型身份变化时应停止，而不是修改校验值绕过。

## 1. 安装与无模型离线检查

安装 Git、curl、uv；[uv 安装文档](https://docs.astral.sh/uv/getting-started/installation/)。Python 3.12 可由 [uv 管理安装](https://docs.astral.sh/uv/guides/install-python/)。项目约束 3.12 次版本，不强制补丁号；严格复现原环境可显式选择 3.12.3。

```powershell
git clone https://github.com/w-harda/RAG.git
cd RAG
# 若未安装 Python，可先 uv python install 3.12.3。
uv sync --locked --python 3.12.3
uv run python --version
uv run pytest -q
uv run python scripts/validate_manifest.py
uv run python scripts/validate_embedding_results.py
uv run python scripts/validate_vectorstore_results.py
uv run python scripts/validate_llm_results.py
uv run python scripts/validate_retrieval_results.py
uv run python scripts/validate_stack_selection.py
uv run python scripts/validate_rag_results.py
uv run python scripts/validate_ablation_results.py
uv build
```

以上不要求报告正文、权重、Ollama 或云密钥，也不调用付费 API。注意 `validate_benchmark.py` 会读取本地 Chunk，必须等第 2 节后再执行。构建包验证 Python 包可发行，不自动把 app/config/data/results 打进 wheel；完整研究复现需要本仓库 checkout。

`uv sync --locked` 会检查锁与项目一致；不要为“安装成功”随意 `uv lock --upgrade` 或替换 Torch/CUDA。跨盘 hardlink 失败后 copy 是正常提示，不影响正确性。模型下载/CPU 推理需要较多存储和内存，最低资源需求未测量，不保证低配机器运行。

## 2. 下载、解析与重建 BGE-M3 缓存

```powershell
uv run python scripts/download_reports.py
uv run python scripts/download_reports.py --verify-only
uv run python scripts/process_corpus.py
uv run python scripts/validate_processed.py
uv run python scripts/validate_benchmark.py
uv run python scripts/run_embedding_benchmark.py --config configs/reproduction_embedding.json --model bge-m3
uv run python scripts/validate_retrieval_results.py --verify-inputs
```

预期：15 报告、491 页、1,297 Chunk、20 题/四类。下载器校验 PDF 文件头、大小、SHA-256，可回退系统 curl；不静默接受官网替换的版本。`process_corpus.py` 默认拒绝覆盖已有 Chunk；仅确定要重建自己的缓存时使用 `--overwrite`，不能改历史数据来改善评分。

首次编码使用公开 `BAAI/bge-m3` 的固定 revision（配置值），可能下载权重且 CPU 编码耗时较长；不是云端生成。`configs/reproduction_embedding.json` 与原 Phase 4 配置仅 `results_directory` 不同：

- 运行向量：`data/processed/embeddings/bge-m3.npz`、`bge-m3.queries.npz`、`bge-m3.metadata.json`。
- 本次编码记录：`data/processed/reproduction/phase4/`，不会覆盖 `results/phase4/`；只跑 BGE-M3 时汇总为 1/3，未选出新胜者是预期现象。
- 后续组件仍读取 Git 中冻结的 Phase 4/5 依据，不把新时间当旧实测值。

最后的 `--verify-inputs` 检查向量字节 SHA-256、Corpus/Query 顺序、Dense/BM25/RRF 候选及来源。**浮点字节相同不是跨硬件保证**；如果不能还原原字节，历史固定管线不允许继续。这是可移植性限制，不是允许更新旧 Phase 5 指纹的理由。完整的新数据版本需要独立重跑/选型/接入，不属于本次收尾。

重新编码默认会写同路径缓存；有重要本地缓存时先保存或使用独立配置/缓存目录。不要直接运行原 `run_embedding_benchmark.py` 或 `run_vectorstore_benchmark.py` 默认入口来启动 UI，它们会覆盖已提交历史实验摘要，并使 Phase 8 指纹失效。

## 3. 准备重排模型、tokenizer 与 RAG 索引

以下使用现代 `hf` CLI，固定 revision；公共模型不需要登录。CLI 参数已按锁定包 `--help` 验证，不使用弃用的 `huggingface-cli`。权重在本机 Hugging Face 缓存，不进入 Git。

```powershell
uv run hf download BAAI/bge-reranker-v2-m3 config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json sentencepiece.bpe.model README.md --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e --max-workers 2
uv run hf cache verify BAAI/bge-reranker-v2-m3 --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
uv run hf download Qwen/Qwen3.5-4B tokenizer.json tokenizer_config.json chat_template.jinja --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --max-workers 2
uv run hf cache verify Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
uv run python scripts/prepare_rag.py --download-tokenizer
uv run python scripts/validate_rag_results.py --verify-inputs
```

部分下载的 `hf cache verify` 可能提示未下载其余文件，这不是已下载文件校验失败；不得忽略校验和不匹配。Qwen 这里只下载 tokenizer，真正生成权重由 Ollama 提供。

`prepare_rag.py` 下载并核验 DeepSeek 官方 tokenizer ZIP/文件哈希（只提取数据，不执行远程示例代码），创建 `data/processed/vectorstores/rag-v1/`。这只是公共文件下载，不读取 API Key、不调用 DeepSeek 生成。即使只用本地 Qwen，固定接入准备仍保留两后端配置与审计要求。

该脚本重算 20 查询 Top-30/融合接入；旧接入记录基础字段不同则拒绝覆盖。索引损坏/不完整不自动删除，诊断后使用新运行配置路径。它可以同内容写回接入 JSON，但不能修改历史实验值；最终验收通过前后字节指纹检查。

## 4. Ollama、真实问答和 Streamlit

### 本地服务

从 [Ollama v0.35.1 官方发布页](https://github.com/ollama/ollama/releases/tag/v0.35.1) 获取适合系统的发行包，不安装其他来源的替代二进制。准备本机服务：

```powershell
ollama --version
# 没有已有服务时，在单独终端启动；不要重复占用同一端口。
ollama serve
# 另一个终端，连接同一服务；首次拉取下载生成权重。
ollama pull qwen3.5:4b
```

固定模型 digest 为 `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`，完整 Provider 身份见 `configs/llm_benchmark.json`。标签是可变的，`ollama pull` 不保证未来仍得到同一 digest；后端发现不同就停止，不悄悄替换。可从 `/api/tags` 核对，禁止直接修改配置 digest 冒充原版本。

默认 endpoint 是 `http://localhost:11434`；若服务使用其他回环端口，应在启动 **Ollama 和问答/Streamlit 各自进程** 前配置相同 `OLLAMA_HOST`，例如 `$env:OLLAMA_HOST = 'http://127.0.0.1:12000'`。只允许本机回环 endpoint，禁用代理。不要停止/升级他人正在使用的服务。

```powershell
# 严格预算预检会加载本地模型/读取原生模板，但不会生成或调用云端 API。
uv run python scripts/prepare_rag.py --check-budgets
uv run python scripts/ask_rag.py "APT44 通常还被称为什么？它被归属于俄罗斯哪个军事情报单位，其行动范围有何特点？" --provider ollama-qwen
uv run streamlit run app/streamlit_app.py
```

打开 `http://127.0.0.1:8501`；默认本地 Qwen。不要把 69 秒级 CPU 全管线响应理解为报错；首次装载还会更长。确认用户/助手消息、Markdown、loading 和真实 Citation。章节为空显示“未标注”，页码是 PDF 物理页码。

端口占用可用 `--server.port 8502`，仍保持回环监听；不要用 `0.0.0.0` 公网暴露这个无认证研究 UI。终端 Ctrl+C 停止自己启动的 Streamlit，不会顺带停止 Ollama。

### 云模式

只读进程环境中的 `DEEPSEEK_API_KEY`，不自动加载 `.env`，也不在 UI 输入/存储密钥。请用本机安全环境变量配置后让启动进程继承；不要把真实密钥放进命令参数、终端历史、文档或 Git。本地使用不需要此变量。

```powershell
# 实际检索/重排/预算，但不访问生成 API、不读取云密钥。
uv run python scripts/ask_rag.py "APT44 的组织背景是什么？" --dry-run
# 有效密钥 + 显式授权；真实生成可能计费。
uv run python scripts/ask_rag.py "APT44 的组织背景是什么？" --allow-cloud
```

Streamlit 选择 DeepSeek 后还需主动勾选授权，未勾选时输入禁用；会话/模型切换及资源释放撤销授权。问题本身也会发送，不得输入敏感信息。无自动重试、自动换模型或隐式付费调用。云 fingerprint/权重不能永远固定，当前服务变化需另做审计，不能保证重复运行同答案。

## 5. 重新实测，而不是复用历史记录

各入口根目录/相对路径规则来自配置；绝不通过删除旧结果重跑“更好答案”。以下仅提供命令，不在最终收尾发起新的付费实验。

### Phase 4 / 5

```powershell
# 三个 Embedding 全部重跑，结果放到 ignored 复现目录；耗时为新运行。
uv run python scripts/run_embedding_benchmark.py --config configs/reproduction_embedding.json
uv run python scripts/validate_embedding_results.py --config configs/reproduction_embedding.json

# 建立单独 Phase 5 配置，不改原配置。使用已核验的同一 BGE-M3 缓存/历史选择。
New-Item -ItemType Directory -Path data/processed/reproduction/configs -Force | Out-Null
$vectorPlan = Get-Content -Raw configs/vectorstore_benchmark.json | ConvertFrom-Json
$vectorPlan.results_directory = 'data/processed/reproduction/phase5'
$vectorPlan.storage_directory = 'data/processed/reproduction/vectorstores'
$vectorPlan | ConvertTo-Json -Depth 20 | Set-Content -Encoding utf8 data/processed/reproduction/configs/vectorstore.json
uv run python scripts/run_vectorstore_benchmark.py --config data/processed/reproduction/configs/vectorstore.json
```

PowerShell 配置写入示例使用 **PowerShell 7**（UTF-8 无 BOM）；Windows PowerShell 5.1 的 `-Encoding utf8` 会写 BOM，与当前 JSON loader 不兼容，可在编辑器保存为 UTF-8 无 BOM。Phase 4/5 增加 `--config` 仅为避免覆盖原实测，默认行为不改变。原 Phase 5 校验/Phase 4–5 绘图脚本默认读取历史目录，未新增统一实验编排器；新 Phase 5 输出已在运行内部校验，人工检查新 JSON，不误用旧路径图冒充新实验图。

### Phase 6 / 7 / 10

复制对应原配置并仅改变以下输出路径（编辑器保存 UTF-8 无 BOM）：

| 阶段 | 必须改变的路径 | 可保留 / 注意 |
| --- | --- | --- |
| 6 | `results_directory`，建议 `results/reproduction/phase6`；建议新 `inputs_path` | benchmark、corpus、Prompt、provider 不变；默认复用已有回答不是重测 |
| 7 | `results_directory`、`reranker_cache_path` | 仍核验原固定向量；不复制旧缓存来冒充新神经打分 |
| 10 | `results_directory`（必须位于 results 内）、`inputs_path`（data/processed 内） | 所有四组共用原 LLM/Prompt；v2 review config 默认不变 |

```powershell
uv run python scripts/run_llm_benchmark.py --config path/to/new-llm.json --prepare-only
# 以下首次云生成可能计费；Phase 6 CLI 没有 --allow-cloud 参数，选择 provider 本身就是运行授权。
uv run python scripts/run_llm_benchmark.py --config path/to/new-llm.json --provider ollama
uv run python scripts/run_llm_benchmark.py --config path/to/new-llm.json --provider deepseek
uv run python scripts/run_llm_benchmark.py --config path/to/new-llm.json --provider all
uv run python scripts/validate_llm_results.py --config path/to/new-llm.json --verify-inputs

uv run python scripts/run_retrieval_benchmark.py --config path/to/new-retrieval.json
uv run python scripts/validate_retrieval_results.py --config path/to/new-retrieval.json --verify-inputs

uv run python scripts/run_ablation_benchmark.py --config path/to/new-ablation.json --prepare-only
uv run python scripts/run_ablation_benchmark.py --config path/to/new-ablation.json --allow-cloud
uv run python scripts/validate_ablation_results.py --config path/to/new-ablation.json --verify-inputs
```

`path/to/new-*.json` 是必须自行建立的占位路径，不能原样运行。新 Phase 6 只有原始回答时语义复核仍待完成，不能复制旧 review 绑定新答案。Phase 10 默认 `ablation_review_v2.json`，历史生成配置中的 4096-token pilot 参数不是正式 v2；80 答 / 80 评审真实重新调用会产生费用。

Phase 7 只算检索排名，不调用 LLM；Phase 10 复用验证过的 Phase 7 重排分数，只记录生成耗时，不伪装本轮神经重排。Phase 9 正常问答则每次真实重排，没有用 Ground Truth 选 Context。

## 6. 人工材料、结果汇总与验收

```powershell
uv run python scripts/prepare_human_review.py --include-text
uv run pytest -q
uv build
git diff --check
git status --short
```

人工工作包说明见 [human_review.md](human_review.md)，主要实验见 [results_summary.md](results_summary.md)，本次实际验证边界见 [final_delivery.md](final_delivery.md)。没有独立人工参与前，不得把空表或 AI 判断称作最终人工质量验证。
