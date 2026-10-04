# Phase 10：四组生成消融实验

> 本文保留该阶段交付时的历史状态和实测口径；当前最终状态与完整流程见 [README](../README.md)、[复现指南](reproduction.md) 和 [最终实验汇总](results_summary.md)。后续阶段已实现的内容不回写历史实验数据。

本阶段比较相同 20 道问题的 Pure LLM、Dense RAG、Hybrid RAG、Hybrid + Reranker。不是重新选模型、重跑组件性能基准或实现 Web UI。原始生成和辅助评审都保存，独立人工评审仍待完成。

## 实验协议与控制变量

配置为 `configs/ablation_benchmark.json`。固定 Phase 8 选型、Phase 9 配置和 Phase 3 的 `apt-qa-v1.1`，问题与已有标准答案不作修改。

| 组 | 生成 Context | 来源 |
| --- | --- | --- |
| Pure LLM | 空 | 不提供报告、标准答案或 Ground Truth |
| Dense RAG | Dense Top-10 全文 | BGE-M3 固定 Query 向量、FAISS HNSW |
| Hybrid RAG | Hybrid Top-10 全文 | 相同 Dense + BM25，RRF 常数 60、等权 |
| Hybrid + Reranker | 重排 Top-10 全文 | 同一 Hybrid Top-30，BGE-Reranker-v2-m3 |

共同 LLM 为 `deepseek-flash`，关闭 thinking，temperature=0，max_output_tokens=768；生成输入窗口上限 8,192、安全余量 256。所有组共用 `configs/ablation_prompt.json` 的 system/user 模板，只改变报告片段。Prompt 允许无片段时基于知识作答，但禁止编造精确 IOC 或来源；有片段时优先依证据，不把推测当事实。

为什么现在新增共同 Prompt：生产管线原有 Prompt 要求仅依据报告、缺证据则弃答，用于 Pure LLM 会强制空证据弃答。本阶段为四组统一建立可比协议，不修改 Phase 6/9 的历史 Prompt，也不把跨 Prompt 的旧问答指标直接作比较。

先用真实 FAISS/BM25/RRF 重算候选、检查有序 ID 与分数，再复用 Phase 7 已实测的 600 个重排分数。Corpus、Query、模型 revision、候选顺序和文本 SHA-256 必须一致；最终排名从旧原始分数重算校验。没有新跑神经重排，因此本轮记录的是**生成耗时**，不是完整管线耗时。选择缓存是为了固定输入、避免引入不同轮次的模型输入变化，不用于伪造本次检索性能。

Top-10 不截断、压缩或根据答案挑选。Context 长度随策略自然变化，是策略的结果而非额外人为调参。生成只读取问题与检索片段，不读取标准答案、证据等级或必答清单。

使用本地 `random.Random(42)` 打乱问题，再在每题内打乱四组；辅助评审另按固定哈希排序。云端无可用 seed，temperature=0 不保证逐字复现；保存实际后端 fingerprint，变化则停止，不自动换模型。

## 运行与恢复

需要已有 Phase 2 Chunk、Phase 4 固定向量、Phase 7 排名/重排分数、Phase 9 已校验的 FAISS 索引和 DeepSeek 官方 tokenizer。准备步骤见 `rag_pipeline.md`。无需新依赖，不需要 Ollama、本地 LLM、Embedding 或 Reranker 权重。

```powershell
uv sync --locked
# 不需要密钥，不调用 API。
uv run python scripts/run_ablation_benchmark.py --prepare-only
# 读取 DEEPSEEK_API_KEY；80 次生成 + 80 次辅助评审首次运行会计费。
uv run python scripts/run_ablation_benchmark.py --allow-cloud
# 可单独完成生成，稍后用上一条命令续接评审。
uv run python scripts/run_ablation_benchmark.py --allow-cloud --generate-only
```

逐回答写原子检查点；网络失败不自动重试，不覆盖已保存回答。不因空回答、截断或引用不合法而删除样本或重生成“更好答案”。若 Windows 短暂占用文件，仅重试磁盘写入；若最后一次替换失败，下次运行先验证 `.json.tmp` 是相同协议与相同前缀的下一条结果，再恢复，不重发已成功的 API 调用。

辅助评审 JSON 解析、事实原文、虚构依据 ID、引用分母或输出完整性不符合协议时，保留原始评审并停止。原文匹配仅允许移除 Markdown 强调、反引号、双引号/中文引号和空白，不更改数字、限定词或事实文字。仍可能发生“API 响应已返回但机器在任何磁盘写入前崩溃”的不可恢复窗口；恢复前应查日志，不声称 exactly-once 网络语义。

实际首轮试评发现两项问题，均公开保留：模型把提供的其他页面误当共享参考；q014 的逐断言评审触及 4,096-token 输出上限。前者加入保守审计：若依据 ID 真实存在于该回答的 Context 但不在共享参考中，将对应事实判定降为 unverifiable，保存降级次数，不扩大参考或修改原始文本；虚构 ID 仍报错。后者独立建立 `configs/ablation_review_v2.json`，全部 80 答统一用 8,192-token 上限重新评审，不只重试坏样本，也不重跑生成。这是观察到试评失败后的工程修正，不宣称整个评审协议事先注册。首轮 `reviews.json` 两条记录和费用继续保留，正式比较使用 `reviews-v2.json` 与 `summary-v2.json`。生成冻结配置中历史 `review` 字段保留用于审计首轮，实际评审参数由独立 v2 配置及结果中的 `review_protocol` 确认。各入口默认使用 v2，也可通过 `--review-config` 显式选择独立评审配置。

原始输入全文位于被 Git 忽略的 `data/processed/ablation/inputs.json`；公开 `results/phase10/inputs_manifest.json` 只含来源 Metadata、调度、输入指纹与 Token 预算。标准答案/问题集本来就在 Git，不能通过隐藏全文宣称数据匿名。

完成后重复运行默认复用回答/评审；离线验证与绘图无需密钥、模型或服务：

```powershell
uv run python scripts/validate_ablation_results.py
uv run python scripts/plot_ablation_results.py
# 有本地处理缓存时，重建完整生成/评审 messages 与输入指纹。
uv run python scripts/validate_ablation_results.py --verify-inputs
uv run pytest
```

新实测必须复制配置到独立结果目录和输入缓存路径，不修改/覆盖本次记录；重跑 80 个回答和 80 次评审会产生新费用。公开语料和研究问题会发送到 DeepSeek；不能用于敏感或内部报告。

## 辅助评审与指标定义

同一模型家族 DeepSeek 逐答辅助评审，JSON mode、关闭 thinking、temperature=0、v2 输出上限 8,192，评审独立窗口上限 32,768。较大评审窗口用于同时容纳候选片段与参考资料，不是扩大生成 Context。官方只提供 JSON object 模式，不假设存在严格 JSON Schema：返回后由本地校验字段、枚举、原文片段、实际 Chunk ID 和引用顺序。参考：[DeepSeek Chat Completion API](https://api-docs.deepseek.com/api/create-chat-completion/)。

四组评审共享同题的 expected_answer、Phase 6 必答清单以及 Ground Truth 物理页前后各一页的**所有 Chunk**。邻页规则在观察回答之前冻结，只作用于评审，不进入生成 Context，也不新增检索相关性标签。它缓解跨页截断（例如 q019 的勒索提示）和参考范围不足，但双栏解析交错、丢符号等处理缺陷仍存在。标准答案不是证据；遇到资料未涵盖的正确知识，不能直接标为错误。

不向评审发送策略名、检索分数、召回等级或其他组回答，但 Context 长度可能暴露组别，因此只是**隐藏标签**，不是严格双盲。每个必答事实给覆盖判定和原因；每个可核实断言给原文、参考 Chunk ID、支持/矛盾/无法核实判定及原因；每次引用按原始出现顺序逐一判断被引用片段是否真正支持相关断言。程序校验可追溯性，不能证明语义判断或断言拆分没有遗漏。

若 AI 把不存在于 provided_sources 的标签判为 supported 或 unverifiable，机器审计统一改为 unsupported 并记录次数。实际 q008 的部分回答沿用 ATT&CK 软件标识 `[S0183]`，与生产管线保留的 `S数字` 引用命名空间冲突，算引用协议错误而非凭空判为事实幻觉。本阶段保留坏样本、暴露此限制，不修改 Phase 9 的引用规则。审计修正后的统计与 AI 原始评语分开，原始评审仍可逐字查看。

| 指标 | 逐题定义 | 边界 |
| --- | --- | --- |
| Accuracy | supported / (supported + contradicted) | 只对可核实断言；无法核实不纳入分母，需并列展示 Unverifiable Rate |
| Completeness | 全部覆盖的必答事实 / 全部必答事实 | 复合事实必须完整正确覆盖；所有 20 题均纳入 |
| Citation Correctness | 被实际引用片段支持的引用次数 / 全部引用次数 | 不去重；格式合法不代表支持；无引用为 N/A |
| Hallucination Rate | AI 标为 contradicted / 全部事实断言 | **参考范围内 AI 矛盾判定代理值**，不是完整真实幻觉率，也不能保证是下界 |
| Unverifiable Rate | 无法核实的断言 / 全部事实断言 | 不假定这些断言全真或全假 |

默认按题宏平均（macro），并报告有效评分题数；无可核实事实的回答 Accuracy 为 N/A，无事实断言的回答幻觉指标为 N/A，不把拒答误当 100% 正确。Citation 同时给 occurrence 微平均（micro）及引用总次数。Pure 没有提供报告引用来源，引用率 N/A 不能画成真实 0。漏答体现在完整性；未给引用但事实正确不等于幻觉；另记录畸形/未知/未闭合来源标签、空回答和输出截断。

这些指标都来自非独立 AI 辅助评审。既有 `llm_rubric_v1.json` 明确禁止把这种评审当最终消融结论，本阶段继续遵守：可产出**暂定定量比较**，最终 Accuracy / 幻觉率与可信结论仍需独立人工逐条复核，尤其是参考之外的断言。不要把这份辅助结果用于宣称研究已证明某配置优越。

## 实测结果

运行后由 `summary-v2.json` 重算四组、逐题、各类别统计，并生成 `ablation_comparison.png`。费用使用仓库冻结的 2026-10-04 单价、实际 Token/cache-hit 用量估算，不是账单。尚不提供 p 值、显著性或其他语料上的泛化结论：样本只有 20 题且为单次运行，同家族自评偏差尚未排除。

本次保留 80 个真实生成回答、80 个完整 v2 评审和 2 个首轮试评。全部生成/评审使用同一个实际后端 fingerprint `aeb56401ca74e127821c4f9126dcb669`。生成最大预检输入 4,196 tokens；v2 评审最大输入估计 10,675 tokens、实际最大输出 4,111 tokens，均通过各自预算校验。生成没有空回答；q014 的 Hybrid、Hybrid + Reranker 两答触及 768 输出上限，原样保留并纳入评价。

下表是逐题宏平均的 **AI 辅助代理分数**，不是独立人工指标：

| 组 | Accuracy | 完整性 | 引用正确率 | AI 矛盾率代理 | 无法核实率 | Accuracy 有效题数 | 生成中位耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pure LLM | 81.11% | 50.00% | N/A | 5.96% | 66.98% | 15/20 | 2.47 s |
| Dense RAG | 96.46% | 93.33% | 98.42% | 2.71% | 39.47% | 20/20 | 2.27 s |
| Hybrid RAG | 93.54% | 93.33% | 91.44% | 2.17% | 41.00% | 20/20 | 2.09 s |
| Hybrid + Reranker | 100.00% | 100.00% | 99.12% | 0.00% | 39.44% | 19/20 | 2.14 s |

引用微平均分别为 N/A、99.02%、95.43%、99.09%，引用出现次数为 0、204、219、219。依据范围审计降级次数分别为 0、88、95、87。最后一组的 100% Accuracy 排除了无法核实断言及一整题，其 0% AI 矛盾率不证明没有幻觉；不能忽略近 40% 的无法核实率。完整性暂定评分显示 RAG 覆盖高于 Pure，但本轮不能据此宣称 Hybrid 必然优于 Dense、重排显著优越或更快。耗时仅含云端生成，未计重排 CPU 开销和本地组件加载。

实际用量对应的峰值单价估算：生成 $0.087283、v2 评审 $0.331054、首轮试评 $0.013212，合计约 **$0.431548**（USD，非真实账单）。不隐藏失败试评的费用。

### 评审抽查发现的局限

这不是新增的人工金标准，只是工程抽查记录；原始 AI 评分未按“更好结果”重写。

- q006:hybrid：AI 评语把“至少五年”当成了确定的“最长驻留时间”，与必答清单关于“不是已知最长恰好五年”的限定存在冲突。因此本阶段将幻觉指标明确标为 AI 判定代理，不称为已确认幻觉的保守下界。
- q019:dense、q018 的部分回答：评语以共享参考报告含有相关内容，去反驳“当前提供片段未出现该信息”的声明，可能混淆检索 Context 范围和参考报告范围。独立复核应分清合理弃答、错误否认和遗漏，不把所有弃答都计为幻觉。
- q016:pure 等评语：部分“contradicted”的理由只是“参考没有这样称呼”，未必证明明确相反，需复核证据强度与完整性。
- q008:hybrid_reranker：`[S0183]` 是报告中的 ATT&CK 标识，但现有保留命名空间把它作为非法来源标签；机器审计覆写了 AI 的宽松判定。它是引用协议缺陷，不自动证明 Tor 这一事实错误。

正式研究验收应由独立评审员逐项修订这些判断，记录仲裁依据，再对同样的 80 个原始回答重算指标。不能直接把本表当论文最终实验结果。

## 本阶段验收与未完成项

工程验收：80 个真实回答与 80 个原始辅助评审；固定控制变量、完整引用 Metadata、输入/答案/评审指纹；离线重算与本地全文重建验证；检查点/坏引用/N/A/不确定事实/API JSON mode 的自动化测试；可视化与阶段提交。

本次验证：86 项测试全部通过；Phase 5–9 旧结果离线校验通过；`--verify-inputs` 重建全部生成 Context 与评审 messages/Token 预算通过；`uv build` 成功。删除子进程中的 `DEEPSEEK_API_KEY` 后，已完成实验成功复用，不调用模型，所有公开 JSON 文件的字节 SHA-256 均保持不变。

仍未完成：独立人工标注与仲裁、参考之外事实的外部核实、更多问题与多轮稳定性分析。Phase 10 的工程消融和辅助比较不等于已完成最终研究质量验证。Phase 11 Web UI 不在本次实现范围。
