# APT 报告语料清单

Phase 1 建立首版 `apt-reports-v1` 语料清单。仓库记录报告元数据、官方来源和内容指纹，不提交原始 PDF。

## 选择标准

报告同时满足以下条件：

1. 来自 Mandiant、CISA 或联合发布机构的官方页面；
2. 明确分析 APT 组织、国家级攻击活动或相关技术行动；
3. 包含组织背景、TTP、IOC、恶意软件、漏洞利用或缓解措施中的至少一类技术内容；
4. 提供可公开访问的 PDF，能够保留稳定页码；
5. 下载内容能够通过文件大小、PDF 文件头和 SHA-256 三重校验。

首版包含 15 份英文 PDF，其中 Mandiant 6 份、CISA 9 份。报告发布时间覆盖 2013—2024 年，包含中国、俄罗斯、伊朗和朝鲜相关活动。

## Manifest 字段

`data/manifest/reports.json` 中每份报告包含：

- `report_id`：仓库内稳定且唯一的报告标识；
- `title`、`vendor`、`apt_group`、`publish_date`、`language`：基本检索元数据；
- `url`：报告介绍或通告的官方页面；
- `download_url`：实际 PDF 下载地址；
- `local_path`：下载后的仓库相对路径；
- `format`、`file_size`、`sha256`：文件类型与完整性信息；
- `tags`：用于筛选和分析的规范化标签。

`url` 与 `download_url` 分离，是为了同时保留可阅读的发布上下文和可自动获取的原始文档地址。

## 校验 Manifest

```powershell
uv run python scripts/validate_manifest.py
```

该命令不访问网络，负责检查字段、日期、HTTPS URL、路径安全、重复 ID、重复本地路径和 SHA-256 格式。

## 下载与验证报告

下载全部报告：

```powershell
uv run python scripts/download_reports.py
```

下载器优先使用 Python 标准库；如果来源站点拒绝该客户端，会自动回退到系统 `curl`。Git for Windows、现代 Windows 以及常见 Linux/macOS 环境通常已包含 `curl`。

只下载指定报告：

```powershell
uv run python scripts/download_reports.py --report-id mandiant-apt44-2024
```

不访问网络，仅验证已有文件：

```powershell
uv run python scripts/download_reports.py --verify-only
```

重新下载指定报告：

```powershell
uv run python scripts/download_reports.py --report-id cisa-aa24-249a --overwrite
```

如果官方文件内容发生变化，下载器会报告预期值与实际值不一致，不会静默接受新版本。此时应先核实发布方是否更新了文件，再有意识地更新 `file_size` 和 `sha256`。

## 版权与版本控制

原始 PDF 位于 `data/raw/`，该目录已被 `.gitignore` 排除。Git 仓库只保存公开来源链接、必要元数据和内容指纹。使用报告时仍需遵守各发布方的版权与使用条款。

## 当前局限

- 首版语料只包含英文 PDF；
- 来源集中在 Mandiant 和 CISA，不能代表全部安全厂商的写作风格；
- 同一组织可能出现在多份报告中，这是为了保留不同年份和不同技术行动的证据；
- 语料质量和覆盖面会在建立 Benchmark 前再次审查，但 Phase 1 不创建问题或答案。
