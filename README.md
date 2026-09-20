# APT-RAG

基于检索增强生成（RAG）的 APT 组织情报知识问答系统与对比验证工程。

项目以可复现、可比较、可解释为核心，按阶段构建 APT 报告语料库、检索 Benchmark、组件对比实验、完整 RAG Pipeline 和 Web UI。

## 当前进度

Phase 2：PDF 解析、清洗、Chunk 与 Metadata Pipeline。

当前已能够把 15 份公开 APT 技术报告确定性处理为带页码、章节和来源信息的 JSONL Chunk。原始 PDF 和处理文本不进入 Git，Benchmark、模型依赖及 RAG 功能将在对应阶段逐步加入。

## 环境要求

- Git
- uv
- Python 3.12（由 `.python-version` 和 `pyproject.toml` 约束）
- curl（仅在来源站点拒绝 Python 下载客户端时作为自动回退）

## 初始化

```powershell
uv sync
```

`uv` 会创建本地 `.venv`，安装当前项目及开发依赖，并严格按照 `uv.lock` 解析环境。

## 验证

```powershell
uv run python --version
uv run pytest
uv run python scripts/validate_manifest.py
uv run python scripts/process_corpus.py
uv run python scripts/validate_processed.py
```

Python 版本应为 3.12，测试和 manifest 校验应全部通过。

## 获取报告语料

```powershell
uv run python scripts/download_reports.py
uv run python scripts/download_reports.py --verify-only
```

报告默认保存到 `data/raw/`，不会被 Git 跟踪。语料选择标准、字段说明和单份报告下载方式见 [语料文档](docs/corpus.md)。

PDF 解析、清洗、Chunk 配置和 Metadata 字段说明见 [处理流程文档](docs/processing.md)。

## 当前目录

```text
.
├── src/apt_rag/       # Python 包
├── configs/           # 可复现实验配置
├── data/manifest/     # 报告清单与 JSON Schema
├── docs/              # 阶段文档
├── scripts/           # Manifest 校验与报告下载入口
├── tests/             # 自动化测试
├── .env.example       # 环境变量模板（不存放真实密钥）
├── .python-version    # Python 3.12 约束
├── pyproject.toml     # 项目与依赖配置
└── uv.lock            # 依赖锁文件（运行 uv sync 后生成）
```

原始报告文件默认由 `.gitignore` 排除。项目使用 manifest、官方来源 URL、文件大小和 SHA-256 校验值保证数据可追溯，并在来源文件变化时显式报错。
