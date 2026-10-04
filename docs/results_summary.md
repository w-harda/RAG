# Phase 4–10 最终实验汇总

本文件只汇总已有记录，**不重测、不更改 JSON/图表，不新增人工结论**。总体为 `apt-reports-v1` 的 15 英文报告、1,297 Chunk，`apt-qa-v1.1` 的 20 中文问题；每类 5 题。原实验日期为 2026-10-04。指标与计时口径不同时不能横向合并。

## Phase 4：Embedding（同语料/分块/问题/256 Token 预算）

| 模型 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | 全排名 MRR | nDCG@10 | Corpus 编码 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BGE-M3 | 0.3750 | 0.5167 | 0.7000 | 0.9000 | 0.5744 | 0.6461 | 607.96 |
| m3e-base | 0.0750 | 0.2417 | 0.4417 | 0.6750 | 0.2864 | 0.3532 | 178.75 |
| text2vec-large-chinese | 0.1250 | 0.2167 | 0.2667 | 0.4167 | 0.2465 | 0.2593 | 628.17 |

按 MRR、再 nDCG 规则选择 BGE-M3。统一 Token 上限不代表统一文本覆盖，不同 tokenizer/池化属于模型变量；没有测其完整长上下文上限。MRR 使用全 1,297 排名，不能直接与后续 MRR@10 对比。离线脚本从 Top-10 重算 Recall/nDCG、用保存的首相关排名核验全排名 MRR 摘要，不假称重跑全部模型/完整排名。

来源：[摘要](../results/phase4/summary.json)、[图](../results/phase4/comparison.png)、[实验协议](embedding_benchmark.md)。

## Phase 5：向量库（固定 BGE-M3 Corpus/Query 字节）

| 后端 | 向量数 | 建库均值 s | 无过滤 p95 ms | 过滤 p95 ms | ANN Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FAISS | 250 | 0.0284 | 0.3686 | 2.6607 | 1.0000 |
| Chroma | 250 | 0.5903 | 4.5253 | 5.7794 | 1.0000 |
| FAISS | 750 | 0.0999 | 0.4384 | 6.6095 | 1.0000 |
| Chroma | 750 | 1.2898 | 4.9940 | 6.7576 | 1.0000 |
| FAISS | 1297 | 0.1920 | 0.3800 | 9.0902 | 1.0000 |
| Chroma | 1297 | 1.8871 | 3.8323 | 7.3665 | 1.0000 |

同 HNSW 公共参数、同插入顺序、每组建库三遍/预热后查询；实现的内部图/持久化机制不等同。完整语料的两库 Evidence Recall@10=0.9000、MRR@10=0.5684、nDCG@10=0.6461。ANN Recall 只表示与精确向量 Top-K 一致，不是证据覆盖。

FAISS 原生不管理 Metadata，项目用 JSON 映射；过滤扫描后构造临时精确索引，Chroma 用原生 where，所以过滤比较是两方案实现代价，不是同算法耗时。索引重载与 1,297 条 Metadata 均通过；磁盘目录均值约 5.85 / 7.40 MiB。不能从 250–1,297 推断百万级或并发性能。

来源：[原始样本](../results/phase5/runs.json)、[摘要](../results/phase5/summary.json)、[图](../results/phase5/comparison.png)、[协议](vectorstore_benchmark.md)。

## Phase 6：LLM（相同标准证据 Context，不是生产检索）

| 后端 | 真实回答数 | 生成中位 s | p95 s | 必答事实辅助覆盖 | 辅助引文支持 | 含无证据断言回答比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ollama qwen3.5:4b | 20 | 3.99 | 14.93 | 79.58% | 85.87% | 30.00% |
| DeepSeek deepseek-flash | 20 | 1.44 | 2.52 | 95.00% | 99.29% | 0.00% |

语义列为**非独立、暂定的证据辅助复核**，不是人工 Accuracy / 全量幻觉率；“含无证据断言回答比例”与 Phase 10 逐断言分母不同，不能混比。相同完整 messages/temperature=0/输出 768，但原生 tokenizer 的实际 usage 不同。Qwen 首次请求约 89.82 s，已包含在 p95/总时长中；这不是稳定生产延迟。云成功回答估算 $0.008502，不是实际账单；本地硬件/能源费用未测。

来源：[摘要](../results/phase6/summary.json)、[图](../results/phase6/llm_comparison.png)、[协议及数据缺陷](llm_benchmark.md)。

## Phase 7：检索策略（无 LLM 变量）

| 策略 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.3750 | 0.5167 | 0.7000 | 0.9000 | 0.5684 | 0.6461 |
| Hybrid | 0.2000 | 0.5583 | 0.7250 | 0.9000 | 0.4476 | 0.5525 |
| Hybrid + Reranker | 0.6500 | 0.8833 | 1.0000 | 1.0000 | 0.8458 | 0.8755 |

候选召回三组均为 1.0000；Hybrid/重排组共享 Top-30。Dense Top-30 已含所有已标注证据，所以本次没有候选召回上的 BM25 收益。Hybrid 前列排序下降，不能预设融合更好。重排改善固定 qrels 排序，但没测 Dense + Reranker，不能证明 BM25 必需。

阶段和单题中位耗时为 0.570 / 1.054 / 31,154.016 ms；不含 Query 编码/索引/模型加载。600 对重排实际打分约 652.92 s，10 对有文档截断。Recall@5=1 不等于全文知识完备、回答正确或零幻觉。

来源：[原始候选/logits](../results/phase7/rankings.json)、[摘要](../results/phase7/summary.json)、[图](../results/phase7/retrieval_comparison.png)、[协议](retrieval_benchmark.md)。

## Phase 8 / 9：选型与接入（不是新质量排名）

质量优先选择 BGE-M3 + FAISS + Hybrid/RRF + BGE-Reranker-v2-m3 + DeepSeek；Qwen 为显式本地方案。选择是观察组件结果后的工程取舍，不冒充事前注册。

Phase 8 JSON 的 `selected_not_integrated` 保留当时历史状态；Phase 9 接通状态来自独立结果，不为更新 README 改掉冻结决策。20 固定查询 FAISS 有序 Top-30/融合均与精确参考及 Phase 7 一致；两 LLM 完整 Context 预算通过（预检最大输入云 4,184 / 本地 4,750）。不保证所有新查询也精确一致。

| Phase 9 q001 真实冒烟 | 本地 Qwen | DeepSeek |
| --- | ---: | ---: |
| 实际输入/输出 tokens | 3057 / 127 | 2933 / 329 |
| LLM 请求墙钟 s | 9.076 | 2.207 |
| 全管线墙钟 s（不含初始化） | 74.908 | 55.016 |
| 真实引用来源数 | 2 | 3 |

每次实际编码/重排/生成，没有 oracle 或答案缓存；两条不足以证明质量、速度优势或平均成本。

来源：[选型](../configs/final_stack.json)、[Top-30/预算](../results/phase9/retrieval_integration.json)、[管线与冒烟文件说明](rag_pipeline.md)。

## Phase 10：四组生成消融（AI Judge，待人工）

共同问题、LLM `deepseek-flash`、Prompt、温度 0、768 输出/8192 共享窗口；只改变 Context 策略。使用验证过的 Phase 7 重排缓存，所以计时仅云生成、不是重新测全管线。正式 v2 统一评审 80 答，另保留 pilot 失败试评和费用。

| 组 | Accuracy | 完整性 | 引用正确率（macro） | AI 矛盾率代理 | 无法核实率 | Accuracy 有效题数 | 生成中位 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure LLM | 81.11% | 50.00% | N/A | 5.96% | 66.98% | 15/20 | 2.47 |
| Dense RAG | 96.46% | 93.33% | 98.42% | 2.71% | 39.47% | 20/20 | 2.27 |
| Hybrid RAG | 93.54% | 93.33% | 91.44% | 2.17% | 41.00% | 20/20 | 2.09 |
| Hybrid + Reranker | 100.00% | 100.00% | 99.12% | 0.00% | 39.44% | 19/20 | 2.14 |

Accuracy = supported / (supported + contradicted)，排除 unverifiable；完整性按必答项全部覆盖；Citation 按原文每次引用，不去重；Hallucination 列仅 AI 在参考范围内判矛盾 / 所有断言，不是全量真实幻觉，不能保证是下界。无有效分母为 N/A，Pure 引用 N/A 不是 0。

引用 micro 为 N/A / 99.02% / 95.43% / 99.09%，出现次数 0 / 204 / 219 / 219。q014 的 Hybrid / 重排回答触及输出上限，原样纳入。最后一组近 40% 断言不可核实，且 Accuracy 排除一题，不能用 100% / 0% 宣称完美。已发现限定词、上下文范围、证据强度误判；同家族自评及试评后调整未排除偏差，无显著性检验。

生成估算 $0.087283、v2 评审 $0.331054、pilot $0.013212，共约 $0.431548；冻结峰值单价和实际 usage 的历史估算，非实际账单。不能把 Phase 6 oracle 费用或单条速度搬到最终系统。

来源：[80 答](../results/phase10/answers.json)、[80 原始 v2 评审](../results/phase10/reviews-v2.json)、[摘要](../results/phase10/summary-v2.json)、[图](../results/phase10/ablation_comparison.png)、[完整协议/误判](ablation_benchmark.md)。

## 最终可支持的结论与未完成项

可以确认工程链路、来源映射、冻结输入、可审计结果与离线统计；当前固定 Benchmark 中 BGE-M3 和重排配置的检索排名表现较好，同时代价较高。不能确认全局模型优劣、因果必要性、生产性能、独立人工 Accuracy 或真实零幻觉。

最终人工研究结论仍待 [独立人工复核](human_review.md)；材料已准备，评分留空。数据修订、更大测试集、多硬件/负载重复实验、Dense + Reranker 对照、云权重变化审计均未在收尾中新增。
