# APT QA Benchmark

Phase 3 建立 `apt-qa-v1`，用于在生成式问答系统实现之前固定检索评测问题和 Ground Truth Evidence，防止根据实验结果反向挑选题目。

Phase 4 证据复核修订为 `apt-qa-v1.1`：为 q012 的 SNMP 默认/弱 community string 和 q018 的 GUID 文件名补充原始 Chunk 引用。题目与答案未改变；三模型全部使用同一修订版重新计算指标。文件名保持 `apt_qa_v1.json`，版本以内部 `benchmark_id` 为准。

## 设计范围

- 20 个中文问题，检索目标为英文 APT 报告，测试跨语言检索能力。
- `background`、`ttp`、`attack_technique`、`ioc` 四类各 5 题。
- 难度覆盖 `easy`、`medium`、`hard`。
- 每题包含人工整理的中文标准答案，但 Embedding 与检索实验只使用问题和证据相关性，不用 LLM 回答评判模型。
- 每条证据指向 Phase 2 生成的真实 `report_id`、PDF 页码和 `chunk_id`。

Benchmark 位于 `data/benchmark/apt_qa_v1.json`，字段契约位于 `data/benchmark/apt_qa.schema.json`。

## Evidence 相关性

`relevance` 是用于 nDCG 的分级相关性：

- `3`：直接回答问题的核心证据；
- `2`：回答问题所需的补充证据；
- `1`：与答案相关但不是必需的背景证据。

Recall@K 和 MRR 将所有已标注 Evidence Chunk 视为相关文档；nDCG 使用上述等级。具体指标实现属于 Phase 4。

当前标注是最小支持证据集合，并非全文穷尽相关性标注；相邻重叠 Chunk 可能也能回答问题而未标注。因此这些指标是对固定 qrels 的初步检索评估，不能视为组织知识覆盖率或答案正确率。20 题可用于工程验证，正式科研结论还需扩充题量、独立人工复核和独立测试集。归因陈述代表相应报告发布时的判断。

## 校验

先完成 Phase 1 下载和 Phase 2 处理，然后运行：

```powershell
uv run python scripts/validate_benchmark.py
```

校验会检查：

1. 顶层、问题和 Evidence 字段契约；
2. 问题 ID 连续且问题文本不重复；
3. 四类问题均被覆盖且总数不少于 20；
4. `report_id` 存在于报告 manifest；
5. 每个 `chunk_id` 存在于本地处理结果，且页码和报告 ID 与 Chunk Metadata 一致。

最后一项要求本地存在 `data/processed/chunks/`。该目录由可复现处理脚本生成，不提交 Git。

## 当前边界

本阶段固定问题、标准答案和证据，不实现向量化、检索、指标计算或 LLM 评分。Phase 4 才会在固定 Corpus、Chunk、Query 和 Top-K 下比较 Embedding 模型。
