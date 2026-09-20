# PDF 解析、清洗与 Chunk Pipeline

Phase 2 将 Phase 1 的原始 PDF 转换为可追溯、可重复生成的 JSONL Chunk。处理结果包含报告级来源信息和 PDF 物理页码，为后续 Benchmark Ground Truth 与 Citation 提供稳定锚点。

## 处理流程

```text
Manifest + PDF
→ SHA-256 校验
→ pdfplumber 逐页抽取
→ Unicode 与空白规范化
→ 重复页眉/页脚识别和移除
→ 跨行断词拼接（保留原文可见连字符）
→ 保守章节识别
→ 页内、章节感知字符切分
→ JSONL + 处理摘要
```

## 关键决策

### 不跨页切分

每个 Chunk 只属于一个 PDF 物理页。这样会牺牲少量跨页上下文，但能够保证 `page` 字段无歧义，后续 Citation 不需要猜测页码。

### 字符切分而不是模型 Token 切分

Phase 4 将比较不同 Embedding 模型。若现在绑定某个 tokenizer，会把 Embedding 模型差异混入 Chunk 变量。固定字符上限能在模型对比时保持同一批 Chunk。

### 页眉页脚按报告统计

清洗器只检查每页顶部和底部的有限行，并且只有某一规范化模板在足够多页面重复时才移除。这比硬编码厂商名称或页码格式更容易复现，也降低误删正文的风险。

### 原始 PDF 和处理文本均不提交

PDF 受各发布方版权约束，处理文本又能由 manifest、固定配置和源文件哈希确定性重建，因此 `data/raw/` 与 `data/processed/` 都被 Git 忽略。

## 固定配置

配置位于 `configs/processing.json`，当前值为：

- 抽取器：`pdfplumber`
- `x_tolerance`：3
- `y_tolerance`：3
- 字符去重 tolerance：1
- 页眉页脚扫描：顶部和底部各 3 行
- 重复阈值：30% 页面，且至少 3 页
- Chunk 大小：1200 字符
- Chunk 重叠：200 字符
- 短 Chunk 合并阈值：200 字符（不跨页且不突破 1200 字符）

修改这些值会改变 Chunk 数据集，开展对比实验后不应在未记录新版本的情况下修改。

## 运行

确保 Phase 1 报告已经下载并验证：

```powershell
uv run python scripts/download_reports.py --verify-only
```

首次处理：

```powershell
uv run python scripts/process_corpus.py
```

明确覆盖已有处理结果：

```powershell
uv run python scripts/process_corpus.py --overwrite
```

校验全部 JSONL、来源 Metadata、Chunk ID、页内序号、字符数和哈希：

```powershell
uv run python scripts/validate_processed.py
```

输出位于：

```text
data/processed/chunks/
├── <report_id>.jsonl
└── summary.json
```

## Chunk 字段

每行是一个独立 JSON object，字段契约见 `data/manifest/chunks.schema.json`。核心字段包括：

- `chunk_id`：由 `report_id + page + 页内序号` 确定；
- `report_id`、`document_title`、`vendor`、`apt_group`；
- `page`：从 1 开始的 PDF 物理页码；
- `section`：保守规则识别的最近章节标题，无法识别时退回报告标题；
- `source_url`：manifest 中的官方发布页面；
- `text`、`char_count`、`text_sha256`。

## 已知边界

- 当前只处理带文本层的 PDF，不执行 OCR；
- 复杂多栏、表格和图中文字可能无法完全保持视觉阅读顺序；
- `section` 是确定性启发式结果，不等同于人工标注；
- 当某一物理页只有章节标题且无法在页内合并时，可能保留少量短 Chunk；
- 图片、图表关系和表格结构化抽取不属于 Phase 2。

## Phase 2 验收基线

在 `apt-reports-v1` 和默认配置下：

- 15 份报告，共 491 个 PDF 物理页；
- 生成 1297 个 Chunk；
- Chunk 长度中位数 1011 字符，平均值约 840 字符；
- 90% 分位数 1198 字符，最大值 1200 字符；
- 32 个 Chunk 少于 100 字符，均受“不跨页”约束保留，主要来自章节或附录扉页；
- Chunk ID、Metadata、字符数、文本哈希、JSONL 文件哈希和页内序号全部通过校验。
