# Phase 7：Dense、Hybrid 与 Hybrid + Reranker

> 本文保留该阶段交付时的历史状态和实测口径；当前最终状态与完整流程见 [README](../README.md)、[复现指南](reproduction.md) 和 [最终实验汇总](results_summary.md)。后续阶段已实现的内容不回写历史实验数据。

本阶段只比较检索排名，不调用 LLM，不实现最终 RAG、Citation 渲染、Web UI，也不替 Phase 8 决定最终技术栈。问题、语料和已有向量不变。

## 核心问题与最小实现

需要区分三个问题：Dense 能否找到证据；BM25 融合是否改善候选召回/前列排序；固定候选池上增加 Reranker 是否改善 Top-K。

最小实现包括框架无关的 `Retriever` / `Reranker` 协议、两种基础检索器、纯 RRF 融合函数，以及可复现实验编排。排序项仅含 Chunk ID 和分数；来源来自真实 Chunk Metadata，不由模型生成。它们现在实现是因为本阶段必须替换检索组件，最终生成、Context 拼装及引用渲染留到后续阶段。

Embedding 和检索质量使用固定 qrels 评估，不以生成答案判断优劣。三组均不调用 LLM，所以不存在 LLM 或 Prompt 的变化；回答质量对检索策略的影响留给 Phase 10 的端到端消融实验。

## 固定输入与实验配置

`configs/retrieval_benchmark.json` 固定以下内容，参数在真实运行前写定，不根据结果调整：

- 语料：相同 15 份报告、1,297 个 Chunk；沿用 Phase 2 的清洗与分块。
- 问题：`apt-qa-v1.1` 的相同 20 题及 graded relevance。
- Embedding：Phase 4 的 BGE-M3 固定 revision、归一化 float32、max_seq_length=256。
- 语料及问题向量：复用同一缓存，检查 ID 顺序、模型/语料/问题指纹、Phase 5 的向量字节 SHA-256，并还原 Phase 4 的 Dense Top-10 和分数。
- Dense 后端：固定 NumPy 精确余弦点积，不混入 FAISS/Chroma ANN 索引误差。这是实验基线，不是提前选定最终向量数据库。
- 查询：原始中文问题，不翻译、不改写、不扩展，不使用 expected_answer 进行查询或重排。
- 每路候选上限 30，Hybrid 融合后保留 30；最终返回 Top-10，评估 K 为 1/3/5/10。
- 所有并列分数按相同的原始语料顺序决定，候选去重。

| 策略 | 候选构造 | 最终排序 |
| --- | --- | --- |
| Dense | Dense Top-30 | 余弦分数 Top-10 |
| Hybrid | Dense Top-30 + BM25 正分 Top-30，RRF 保留 Top-30 | RRF Top-10 |
| Hybrid + Reranker | **完全相同的 Hybrid Top-30** | BGE 重排 Top-10 |

Dense 与 Hybrid 的输出预算一致，但 Hybrid 额外执行一个词法分支，最多对 60 个去重候选进行融合。这属于检索策略变量本身，不能将它称为相同计算预算。Hybrid 与 Hybrid + Reranker 的候选池逐题完全一致；Reranker 无法找回池外证据。

## BM25 与跨语言限制

使用小型、可单元测试的 BM25 实现，不新增搜索服务或分词依赖：

```text
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
score(q,d) = sum_t idf(t) * tf(t,d) * (k1+1)
             / (tf(t,d) + k1 * (1-b + b*len(d)/avg_len))
```

`k1=1.2`、`b=0.75`，查询词去重，忽略查询词频。正 IDF 采用 Lucene 文档中的形式，但不是运行 Lucene，也不声称与其完整分析器/分数缩放完全相同。[Lucene BM25 文档](https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/search/similarities/BM25Similarity.html)

分词规则为 `ascii_identifiers_cjk_chars_v1`：英文 casefold，ASCII 标识符保留内部连字符/下划线，中文逐字；例如 CVE-2017-6742 保持为一个标识符。标点分割，无停用词过滤或词干化，只索引 Chunk 原文，不给标题/标签等 Metadata 加权。

由于问题是中文、报告主要是英文，BM25 主要利用共有的 APT 名称、工具名和 IOC，不能跨语言匹配一般语义。只有正分候选参与融合；完全没有词法命中时，Hybrid 回落到 Dense 顺序，而不是把任意零分文档灌入 RRF。本阶段不通过翻译改善 BM25，以免引入新的模型/提示词变量。

## RRF 与固定版本 Reranker

RRF 不直接相加不同尺度的余弦与 BM25 分数，而使用排名：

```text
RRF(d) = 1 / (60 + dense_rank(d)) + 1 / (60 + bm25_rank(d))
```

排名从 1 开始，缺失分支贡献为 0，两分支等权。60 是预先固定的排名常数，并非在 20 题上调优。[RRF 官方说明](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)

Reranker 使用 `BAAI/bge-reranker-v2-m3`（Apache-2.0），固定 revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`。按官方模型卡用分类模型对 `(原始问题, 原始 Chunk 文本)` 输出一个相关性 logit；它不是 LLM 生成模型，也不接受标准答案或 relevance 标签。[官方模型卡](https://huggingface.co/BAAI/bge-reranker-v2-m3)

采用当前 Transformers 的 `dtype` 参数，禁用远程代码执行，固定 CPU/float32、batch_size=4、Torch intra-op threads=4、seed=42 和确定性算法。当前项目已有 CPU Torch，本阶段不升级/替换为 CUDA 安装。仅将直接使用的 Torch/Transformers 声明为直接依赖，锁文件中版本未变化。

模型输入最大 512 Token，`truncation=only_second`，保留整个问题，只截断文档。每个候选对保存截断前/后的 Token 计数和原始 logit；logit 不当作校准后的概率。Embedding 原有输入上限仍为 256，本实验比较的是增加这一 512-Token 重排阶段的整体效果，不是同等计算预算的模型容量比较。[Tokenizer 参数](https://huggingface.co/docs/transformers/main_classes/tokenizer)

## 下载、运行与复现

使用 `hugging-face:hf-cli` 技能：通过 `hf models info` 确认 commit/license，再用现代 `hf download` 下载所需文件，使用 `hf cache verify` 校验。只下载 7 个运行/说明文件；未下载的网页配图及 `.gitattributes` 不参与模型推理。权重留在本机 Hugging Face 缓存，不进入 Git。

```powershell
uv sync --locked
uv run hf download BAAI/bge-reranker-v2-m3 config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json sentencepiece.bpe.model README.md --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e --max-workers 2
uv run hf cache verify BAAI/bge-reranker-v2-m3 --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
uv run python scripts/run_retrieval_benchmark.py
```

运行要求 Phase 2 Chunk、Phase 4 固定向量缓存及 Phase 5 指纹记录已存在。配置默认 `local_files_only=true`，缺少模型时明确失败，不静默下载另一 revision。公共模型不要求 API Key；本阶段不读取 DeepSeek/OpenAI 密钥，不调用云端推理或付费 Jobs。

逐题模型打分保存到被 Git 忽略的 `data/processed/reranking/phase7.json`。问题、候选顺序和文本指纹变化时拒绝继续旧检查点。续跑复用已完成的候选打分及原始耗时；未完成题目才执行模型推理。正式输出是 `results/phase7/rankings.json` 与 `summary.json`。

完成后再次运行只验证和复用已有结果，不重新加载模型。若要真正重测，复制配置，修改 **results_directory 和 reranker_cache_path 两个路径**，再用 `--config` 指定新文件，避免覆盖本次记录或将缓存耗时冒充新实测。

离线复核无需报告正文、Embedding 缓存、模型或密钥：

```powershell
uv run python scripts/validate_retrieval_results.py
uv run python scripts/plot_retrieval_results.py
uv run pytest -q
```

离线校验会从保存的分支候选重算 RRF，从保存的 logits 重算重排顺序，从固定 qrels 重算各策略/分类指标，并检查报告 manifest 的标题、厂商与 URL。它验证统计和输入关系，不声称重新运行神经网络。

如有本地 Chunk/向量，可以进一步验证完整输入，不加载 Reranker：

```powershell
uv run python scripts/validate_retrieval_results.py --verify-inputs
```

该命令检查实际语料、向量字节、Dense/BM25 候选、重排输入指纹及真实页码/section Metadata。`--write-summary` 可显式重建统计摘要，不改原始排名；三个入口均支持 `--config`。

## 指标口径与当前边界

- Recall@1/3/5/10：固定 evidence 集合的覆盖比例，再对问题宏平均。
- MRR@10：只在最终 Top-10 中找首个相关 Chunk，超出 10 记 0；不能与 Phase 4 的全排名 MRR 直接混比。
- nDCG@10：沿用 relevance=1/2/3 的分级相关性。
- candidate_recall：排序前候选池的证据覆盖，Dense 用其 Top-30，另外两组共用 Hybrid Top-30。这是当前候选池约束下的召回上界。
- 分别保存四个问题类别的指标和逐题结果，不能仅凭总平均推断 IOC/TTP 表现一致。
- 延迟来自单遍各阶段的真实墙钟耗时，策略时延为共同 Dense + 需要的 BM25/RRF/Reranker 分量之和；不包含报告/向量读取、Embedding 编码、BM25 建索引或 Reranker 加载。索引/加载时间另记。

没有独立重复负载实验、专用隔离机器或云/GPU Reranker 对比；本机其他活动未完全隔离。中位/p95 是 20 个不同问题的描述统计，不是稳定生产延迟或统计显著性结论。候选缓存复用时保留首次打分耗时并标记 reused。

qrels 是最小支持证据，不是全部相关 Chunk，重叠或邻页即使能回答也可能未标注。Phase 6 已发现的 PDF 双栏交错、路径符号和证据完备性问题仍保留不变；不得看到结果后修改标签来提高某个配置分数。后续数据修订必须版本化并重跑受影响实验。

正式研究结论还需更多问题、独立相关性标注、独立测试集和重复实验。本阶段不评估回答 Accuracy、Completeness、最终 Citation Correctness 或 Hallucination Rate，这些需要后续生成与消融实验。

## 真实结果（2026-10-04）

| 策略 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.3750 | 0.5167 | 0.7000 | 0.9000 | 0.5684 | 0.6461 |
| Hybrid | 0.2000 | 0.5583 | 0.7250 | 0.9000 | 0.4476 | 0.5525 |
| Hybrid + Reranker | 0.6500 | 0.8833 | 1.0000 | 1.0000 | 0.8458 | 0.8755 |

三个候选池的 candidate_recall 均为 1.0000；Hybrid 与重排组使用完全相同的候选。本次没有发现 BM25 增加候选召回的收益，Dense Top-30 已覆盖全部已标注证据；改进主要体现在重排后的前列证据排序。**现有三组不能证明 BM25 对重排后的提升是必需的**，没有另外测试 Dense + Reranker，不补作该结论。

等权 Hybrid 的前列排序反而下降：MRR@10 比 Dense 低约 0.1208，nDCG@10 低约 0.0935；不能预设 Hybrid 一定更好。重排组相较 Hybrid 的 MRR@10 高约 0.3982，但只代表这 20 题的固定 qrels 结果。Recall@5=100% 不表示真实场景中覆盖全部相关知识或回答无幻觉。

四类问题的 MRR@10：

| 类别（各 5 题） | Dense | Hybrid | Hybrid + Reranker |
| --- | ---: | ---: | ---: |
| 组织背景 | 0.6583 | 0.3352 | 0.7167 |
| TTP | 0.4000 | 0.2733 | 0.9000 |
| 攻击技术 | 0.5867 | 0.5867 | 0.9000 |
| IOC | 0.6286 | 0.5952 | 0.8667 |

Hybrid 的 IOC Recall@10 从 Dense 的 0.8 升到 1.0，但组织背景从 1.0 降到 0.8；总平均 0.9 掩盖了这种差异。

| 描述性性能统计 | Dense | Hybrid | Hybrid + Reranker |
| --- | ---: | ---: | ---: |
| 阶段和单题中位耗时 | 0.570 ms | 1.054 ms | 31,154.016 ms |
| 阶段和单题 p95 耗时 | 0.906 ms | 1.542 ms | 40,483.546 ms |

Reranker 对 600 对输入全部实际推理，没有复用旧分数；总打分约 652.92 秒，模型初始化/加载（包括首次导入）约 51.46 秒，BM25 建索引约 0.177 秒。600 对中有 10 对发生文档截断；最长原始输入 700 Token，实际输入最大 512。本机 CPU 与 Phase 6 相同，Torch `2.14.0`、Transformers `5.17.0`、NumPy `2.5.3`、Python `3.12.3`；运行版本记录保存在原始结果中。

CPU 重排提高了本次排名指标，但单题中位额外耗时约 31 秒。它不是直接可用的低延迟交互结论；GPU、批处理或更小模型的收益尚未测量。本阶段只报告该质量/计算成本权衡，不提前确定最终部署配置。

[对比图](../results/phase7/retrieval_comparison.png)采用对数延迟纵轴，避免把数量级差异隐藏；质量轴仍为线性比例。

## 文件变更、验证与未完成事项

新增：

- `configs/retrieval_benchmark.json`：候选预算、BM25/RRF 及 Reranker 固定配置。
- `src/apt_rag/retrieval/__init__.py`、`strategies.py`：接口、Dense、BM25 与 RRF。
- `src/apt_rag/reranking/__init__.py`、`base.py`、`bge.py`：接口与固定版本分类模型适配器。
- `src/apt_rag/evaluation/retrieval_benchmark.py`：输入/缓存校验、真实实验、逐题与分类指标。
- 三个脚本：`run_retrieval_benchmark.py`、`validate_retrieval_results.py`、`plot_retrieval_results.py`。
- `tests/test_retrieval_strategies.py`、本文，以及 `results/phase7/` 的原始候选/分数、摘要和图表。

修改：`README.md` 更新进度；`pyproject.toml`、`uv.lock` 声明直接使用的已有 Torch/Transformers，没有更换旧依赖版本。

验证覆盖数学公式、中文/标识符分词、稳定并列排序、零分过滤、候选预算、重复 ID、非有限分数、相同候选重排、标准答案隔离、检查点复用、MRR 截止口径和已保存真实实验的离线重算。旧阶段功能保持不变。

尚未完成：更大且独立复核的 qrels、独立测试集、隔离环境重复耗时测量、不同 Reranker/GPU 对比、Dense + Reranker 补充对照，以及端到端生成与回答质量评估。没有自动推进 Phase 8。

建议提交说明：`feat: compare dense hybrid and reranked APT retrieval`
