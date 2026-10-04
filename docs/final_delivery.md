# Phase 12 最终交付与验收

## 交付边界

Phase 0–11 工程已集成，Phase 12 只整理 README、架构/结构/配置来源、从 clone 开始的复现、Phase 4–10 总表和独立人工复核材料。没有新增问答核心功能、Agent、MCP、用户系统、多轮记忆或新框架。

**研究质量验收不完全：Phase 10 独立人工复核仍待完成，已完成人工复核 0 条。** AI 原始评分继续明确标为辅助代理，80 个原始回答和所有旧实验数据保留不变，不能将最终工程交付描述为独立人工研究结论已完成。

## 最小改动与理由

- 重整 [README](../README.md)，新增 [复现指南](reproduction.md)、[实验汇总](results_summary.md)、[人工流程](human_review.md)；历史 Phase 4–11 文档增加历史状态提示，解决旧“未进入后续阶段”与当前状态混淆。
- Phase 8 校验 CLI 的状态文案改为“历史选型”，避免把冻结 JSON 的当时状态错误表述为当前未接入；不改校验逻辑或决策 JSON。
- Phase 4/5 CLI 增加 `--config`，不改变默认实验/数学逻辑；独立 `reproduction_embedding.json` 只改变结果输出目录。现在需要它，因为默认 Phase 4 重建会改掉被冻结的历史摘要，造成新用户 UI 准备失败；不是为未来扩展架构。
- 增加 `validate_embedding_results.py`：补齐 Phase 4 离线核验入口，检查 Top-K 指标/首相关排名摘要和选择；不声称从 Top-10 恢复未保存的完整模型排名。
- 增加 `prepare_human_review.py`：可再生 80 答的空白人工工作包及协调映射，正文只在本地 ignored 目录，不自动评分或修改 AI JSON。
- 交付测试验证复现配置仅改输出、指标表与原 JSON 一致、人工材料无自动评分、不覆盖已有表、相对文档链接和独立 CLI 配置入口。

Phase 9 核心 `src/apt_rag/rag/`、所有其他核心模块、原配置/Prompt/Benchmark、依赖及 `results/phase4…10/` 不改。

## 验收命令

以下命令从仓库根目录运行；全部是测试/校验/构建，不调用付费生成。默认不运行会写图/覆盖旧实验的命令。

```powershell
uv sync --locked
uv run pytest -q
uv run python scripts/validate_manifest.py
uv run python scripts/download_reports.py --verify-only
uv run python scripts/validate_processed.py
uv run python scripts/validate_benchmark.py
uv run python scripts/validate_embedding_results.py
uv run python scripts/validate_vectorstore_results.py
uv run python scripts/validate_llm_results.py --verify-inputs
uv run python scripts/validate_retrieval_results.py --verify-inputs
uv run python scripts/validate_stack_selection.py
uv run python scripts/validate_rag_results.py --verify-inputs
uv run python scripts/validate_ablation_results.py --verify-inputs
uv build
git diff --check
git status --short
```

无本地缓存的 clone 可先运行复现指南第 1 节的离线命令，不能假称 `--verify-inputs` 不需要数据。含 inputs 的校验要第 2–4 节准备后才运行，不需要云密钥。

## 本次实际验证记录

验收日期：2026-10-05（Asia/Shanghai），Windows / Python 3.12.3 / CPU。记录的是本次实际执行，不外推任意新机器。

- `uv sync --locked` 成功；最终 **110 项测试通过**（原 98 项 + 12 项交付检查），`uv build` 的 sdist/wheel 成功，`git diff --check` 无空白错误。LF→CRLF 提示不是检查失败。
- Manifest、15 PDF 大小/SHA、491 页/1,297 Chunk、20 题 Evidence 校验通过；Phase 4–10 结果校验全部通过。Phase 6/7/9/10 的 `--verify-inputs` 重建/核验实际 Context、候选、指纹和预算通过，不调用付费 API。
- 收尾前后比对原已跟踪 **47 份**结果/图表、配置、Manifest/Benchmark 和依赖文件的 SHA-256，字节不变；核心 `src/` 和前端 `app/` 无改动。没有重绘旧图、重跑旧评分或改选型依据。
- 独立工作包已真实导出：80 答、逐答和必答项空白表、断言/引用空表头、协调映射；已核验并包含本地正文，所有评分/评审人留空，`completed_human_reviews=0`，`independent_human_review=false`。

### 隔离克隆到实际问答

在 ignored `data/processed/phase12-clone/` 做独立本机克隆：从已提交 Phase 11 树开始，仅覆盖本次待提交的 Phase 4 CLI 和新增复现配置。**没有复制原 PDF、Chunk、向量、索引或本地虚拟环境。** 这是本机隔离 checkout 验证，不冒称另一台硬件或无任何共享下载缓存的环境。

1. 全新 `.venv` 执行 `uv sync --locked`，uv 安装缓存可复用；随后实际从官方 URL 下载全部 15 PDF，通过原 SHA 校验。
2. 实际重新解析得 491 页、1,297 Chunk，再重新执行 BGE-M3 编码（Corpus 约 733.38 s，Query 约 2.35 s；新时间只写 ignored 目录）。Corpus 和 Query 字节均通过原 Phase 5 SHA-256，原 Phase 4 指标也还原；没有复制旧向量或覆盖历史摘要。
3. 重新下载并校验官方 DeepSeek tokenizer，创建新 FAISS 索引，20 题 Top-30/融合与原结果一致。已有本机 Hugging Face 权重/tokenizer 缓存复用，并由 `hf cache verify` 核验 Reranker 7 文件、Qwen tokenizer 3 文件；没有谎称重新下载生成权重。
4. 连接已有本机 Ollama（环境指定回环端口，固定 0.35.1/原模型 digest），实际两后端预算检查通过，输入最大云 4184 / 本地 4750；没有访问 DeepSeek 生成 API。
5. 新 checkout 的原 Phase 9 管线实际回答 q001，`status=answered`、`finish_reason=stop`，输入/输出 3057/127 tokens；本次生成约 7.26 s、管线约 64.17 s、不含初始化，CLI 全程约 96.80 s。两条引用映射到第 4/6 页及真实章节/URL/Chunk ID。这是单条冒烟，不加入原研究结果或作为性能结论。
6. 在该 checkout 实际运行 `uv run streamlit run app/streamlit_app.py --server.port 8502`，健康检查 HTTP 200；Streamlit 官方 AppTest 执行未替换的页面，确认加载配置、本地默认模型、输入启用且无异常。已停止本次临时 8502 服务，不停止用户原 Ollama 或原 8501 服务。

本机验证缓存和问答记录留在上述 ignored 克隆路径供检查，不作为项目安装前提，不提交 Git；其中的新向量是实算的。结果表继续只对应原 2026-10-04 实验，不混入本次新耗时。克隆复现仍依赖固定官方文件可获取、共享模型 revision 可下载、相同字节重建和可用的本地服务，不能许诺未来全部外部条件不变。

## 仍存在的限制

1. 未独立人工标注/仲裁，AI Judge 同模型家族自评、参考范围和断言拆分偏差仍存在。
2. 20 题小样本、非穷尽证据、英文 PDF 来源集中，未证明统计显著性、泛化性或完整 APT 覆盖。
3. 双栏交错、符号/跨页截断和必答项范围缺陷保留；没有收尾时“修数据提分”。
4. 字节指纹/固定服务/调试模板绑定降低跨机器可移植性；云权重/fingerprint、官网 URL、可变 Ollama 标签不能永久固定。
5. CPU 重排开销高，未验证生产 SLA、并发、GPU 提升、更多候选/Reranker 或 Dense + Reranker。
6. Citation 标签与 ATT&CK S编号冲突；来源映射有效不等于语义支持，生成仍可能拒绝或弃答。
7. 费用是历史估算非账单；软件锁/仓库不能包含真实云密钥或保证外部 API 长期可用。
8. 临时会话非持久多轮记忆；本机 UI 无登录/公网生产保障。第三方内容遵守自身许可证，根项目 LICENSE 尚未由所有者明确。

本次交付在 Phase 12 停止，不自动继续功能开发或以 AI 替代独立人类。
