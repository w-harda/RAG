# APT-RAG

基于检索增强生成（RAG）的 APT 组织情报知识问答系统与对比验证工程。

项目以可复现、可比较、可解释为核心，按阶段构建 APT 报告语料库、检索 Benchmark、组件对比实验、完整 RAG Pipeline 和 Web UI。

## 当前进度

Phase 0：项目初始化和开发环境。

当前仅包含可安装的最小 Python 包和环境冒烟测试。APT 报告、解析流程、Benchmark、模型依赖及 RAG 功能将在对应阶段逐步加入。

## 环境要求

- Git
- uv
- Python 3.12（由 `.python-version` 和 `pyproject.toml` 约束）

## 初始化

```powershell
uv sync
```

`uv` 会创建本地 `.venv`，安装当前项目及开发依赖，并严格按照 `uv.lock` 解析环境。

## 验证

```powershell
uv run python --version
uv run pytest
```

Python 版本应为 3.12，测试应全部通过。

## 当前目录

```text
.
├── src/apt_rag/       # Python 包
├── tests/             # 自动化测试
├── .env.example       # 环境变量模板（不存放真实密钥）
├── .python-version    # Python 3.12 约束
├── pyproject.toml     # 项目与依赖配置
└── uv.lock            # 依赖锁文件（运行 uv sync 后生成）
```

原始报告文件默认由 `.gitignore` 排除。后续语料阶段将使用 manifest、来源 URL 和 SHA-256 校验值保证数据可追溯与可复现。
