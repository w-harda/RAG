# Phase 10 独立人工复核：工作包与执行说明

**状态：待独立人工完成。已完成人工复核数量：0。** 本文和导出脚本只是准备材料；AI 不充当独立人类评审员，不替人填写分数，不声称已经仲裁或完成最终研究质量验证。

## 材料入口与原始数据

- `results/phase10/answers.json`：不可修改的 80 个原始回答，包括截断/坏引用。
- `results/phase10/inputs_manifest.json`：每答真实 Context Metadata 和同题共享参考（Ground Truth 页 ±1 的 Chunk）。
- `data/benchmark/apt_qa_v1.json`：问题、标准答案、证据；`llm_rubric_v1.json`：必答事实清单。
- `results/phase10/reviews-v2.json` / `summary-v2.json`：AI 原始评审/暂定统计，**不先发给评分者**，避免锚定。
- `results/phase10/reviews.json`：失败 pilot，仅供协调员审计，不冒充正式人工结果。

```powershell
# 有本地 Chunk 时，核验 Corpus/每项 Metadata/文本 SHA 后加入全文。
uv run python scripts/prepare_human_review.py --include-text
# 无本地 Chunk 时也可导出链接/Metadata 版（任选其一；不能覆盖同一目录）。
uv run python scripts/prepare_human_review.py --output data/processed/human-review-links
```

导出在 `data/processed/` 下且目录已存在时拒绝覆盖，避免丢失人工填写的内容。不调用 API，不读取真实密钥，不更新既有结果。工作包带原 80 答集合及输入指纹，答案指纹沿用 JSON 规范化 SHA-256，不是裸字符串字节哈希。

| 文件 | 用途 |
| --- | --- |
| `review_packet.json` | H001–H080、原问题/答案、必答项、标准答案、提供片段与共享参考、官方网页/PDF URL；不含 AI 分数、策略名 |
| `answer_review.csv` | 80 行 pending，评分/人名/日期/备注留空；填 supported/contradicted/unverifiable 断言数与逐次引用数 |
| `fact_coverage.csv` | 每答每项必答事实预列，coverage/evidence/reason/reviewer 留空 |
| `claims.csv` | 空表头；评审员自行拆分全部可核实事实断言并增行，不沿用 AI 断言划分 |
| `citations.csv` | 空表头；按原文每次出现顺序增行，同标签重复引用不能去重 |
| `coordinator_mapping.json` | 协调员专用：H 编号 → 原回答 ID/策略/答案指纹，不提前给评分者 |

仅 JSON 包含模型原始答案，避免将多行答案直接塞进 CSV 或触发表格公式。包中来源正文留在本机 ignored 目录，不把报告全文提交 Git；人名等个人数据也不自动发布。Git 保存的是可重建脚本和流程，clone 后可重新导出完整材料。

## 人类操作流程

1. 由真实、独立于模型输出的评审员执行；建议两位具备 APT 阅读能力的人独立评分，然后由另一位/共同讨论仲裁。若只有一人，应如实称“单人复核”，不能冒充双盲或双人标注。
2. 协调员分发 packet 与空表，不分发映射/AI Judge。编号按固定哈希打散、策略标签隐藏；Context 数量仍可能透露组别，因此不声称严格双盲。
3. 先按官方 PDF 物理页核实 Ground Truth/必答项，记录文本抽取错误或参考缺失。**标准答案不是证据。** 不以“参考未出现”直接判 contradicted；需要直接相反证据，否则 unverifiable 或追加外部查证。
4. 每答逐条拆分断言、逐必答项判全覆盖/部分/缺失，逐次引用核对真实被引用 Chunk 是否支持关联事实。区分“本次检索片段没有”和“报告全篇没有”，合理弃答不能自动记幻觉。
5. 每项填写 evidence（Chunk ID，或外部来源 URL/标题/物理页/访问日期）、reason、reviewer；逐答补日期/status。对参考外知识另行核实，不能让 AI 代签人类判断。
6. 独立完成后保留各自原表，另建仲裁表，不覆盖某个评审者原意见。记录分歧、证据和仲裁人，并报告一致性统计及方法。
7. 绑定相同 80 答指纹，再独立重算指标；发布到新人工结果目录而非覆写 AI JSON。导出脚本**不包含人工计分/汇总器**；真实表格完成前不得填造最终分数。

## 评分定义与分母

保持 Phase 10 可比较口径；若人类修改定义，要另版报告并说明差异，不与旧 AI 值悄悄混比。

- `claims.csv` 的 judgment：supported / contradicted / unverifiable。逐题 Accuracy = supported / (supported + contradicted)，无可核实断言为 N/A。
- 完整性：必答事实全部正确覆盖才记 1；部分/缺失记 0。每题覆盖数 / 原必答项总数，再对全部 20 题宏平均。
- 引用正确率：支持关联事实的引用出现次数 / 全部引用出现次数；包括未知/非法引用为不支持，不去重，无引用 N/A。宏平均与 occurrence 微平均同时报告。
- Hallucination Rate：contradicted / 所有拆分断言。若仍仅按限定参考核实，须标为“参考范围内人类矛盾判定”，不能称已量出全部真实幻觉；无法核实率同时报告。
- 每组固定 20 答，报告有效分母题数、断言拆分规则、无引用/无事实处理、截断数。拒答不能默认 100% Accuracy；原 q014 两条 `finish_reason=length` 仍纳入。

表格中未填写的格子是**缺失评分，不是 0 或 N/A**；完成前禁止默认填零算平均。只有真实评审员完成后才更新独立人工完成标记。

## 重点核查清单（不是预填判断）

- q006 的“至少五年”与“最长恰好五年”限定差异。
- q018 路径缺 `%`、文件名结构来源和抽取符号；q019 本次 Context 与共享参考范围的区别、勒索提示跨页截断。
- q016:pure 等“参考没有称呼”是否真有相反证据，避免误判矛盾。
- q008 的 ATT&CK `[S0183]` 是保留来源标签协议冲突，不能直接等同事实错误。
- q009/q014/q016/q017 双栏交错，q007/q011 必答项是否超出问题直接要求。
- Hybrid + Reranker 的 AI 100% Accuracy 排除约 39% 不可核实断言及一题分母；不能先假设其质量完美。

这些是旧工程抽查暴露的问题，不是人工金标准。资料修订必须独立版本化，保留原数据与同题同答比较，不为某策略改分或删题。
