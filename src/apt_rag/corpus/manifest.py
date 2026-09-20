"""加载并校验 APT 报告 manifest。"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

REPORT_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
LANGUAGE_PATTERN = re.compile(r"^[a-z]{2}(?:-[A-Z]{2})?$")
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
TAG_PATTERN = REPORT_ID_PATTERN

REQUIRED_REPORT_FIELDS = {
    "report_id",
    "title",
    "vendor",
    "apt_group",
    "publish_date",
    "language",
    "url",
    "download_url",
    "local_path",
    "format",
    "file_size",
    "sha256",
    "tags",
}
ALLOWED_TOP_LEVEL_FIELDS = {"schema_version", "corpus_id", "created_at", "reports"}


class ManifestValidationError(ValueError):
    """manifest 不符合当前语料契约。"""


def _is_https_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def _is_safe_local_pdf_path(value: object) -> bool:
    if not isinstance(value, str) or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and len(path.parts) == 4
        and path.parts[:2] == ("data", "raw")
        and path.suffix == ".pdf"
    )


def validate_manifest(data: object, *, minimum_reports: int = 15) -> dict[str, Any]:
    """校验 manifest，并在失败时一次性返回所有发现的问题。"""

    errors: list[str] = []
    if not isinstance(data, dict):
        raise ManifestValidationError("manifest 顶层必须是 JSON object")

    unknown_top_level = data.keys() - ALLOWED_TOP_LEVEL_FIELDS
    if unknown_top_level:
        errors.append(f"manifest 包含未知字段：{', '.join(sorted(unknown_top_level))}")

    if data.get("schema_version") != "1.0":
        errors.append("schema_version 必须为 1.0")
    if not isinstance(data.get("corpus_id"), str) or not data.get("corpus_id", "").strip():
        errors.append("corpus_id 必须是非空字符串")
    try:
        date.fromisoformat(data.get("created_at", ""))
    except (TypeError, ValueError):
        errors.append("created_at 必须是 YYYY-MM-DD 日期")

    reports = data.get("reports")
    if not isinstance(reports, list):
        errors.append("reports 必须是数组")
        reports = []
    elif len(reports) < minimum_reports:
        errors.append(f"reports 至少需要 {minimum_reports} 项，当前为 {len(reports)} 项")

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for index, report in enumerate(reports):
        prefix = f"reports[{index}]"
        if not isinstance(report, dict):
            errors.append(f"{prefix} 必须是 object")
            continue

        missing = REQUIRED_REPORT_FIELDS - report.keys()
        if missing:
            errors.append(f"{prefix} 缺少字段：{', '.join(sorted(missing))}")
            continue
        unknown = report.keys() - REQUIRED_REPORT_FIELDS
        if unknown:
            errors.append(f"{prefix} 包含未知字段：{', '.join(sorted(unknown))}")

        report_id = report["report_id"]
        if not isinstance(report_id, str) or not REPORT_ID_PATTERN.fullmatch(report_id):
            errors.append(f"{prefix}.report_id 格式无效")
        elif report_id in seen_ids:
            errors.append(f"{prefix}.report_id 重复：{report_id}")
        else:
            seen_ids.add(report_id)

        for field in ("title", "vendor", "apt_group"):
            if not isinstance(report[field], str) or not report[field].strip():
                errors.append(f"{prefix}.{field} 必须是非空字符串")

        try:
            date.fromisoformat(report["publish_date"])
        except (TypeError, ValueError):
            errors.append(f"{prefix}.publish_date 必须是 YYYY-MM-DD 日期")

        language = report["language"]
        if not isinstance(language, str) or not LANGUAGE_PATTERN.fullmatch(language):
            errors.append(f"{prefix}.language 必须使用简化的 BCP 47 格式")

        for field in ("url", "download_url"):
            if not _is_https_url(report[field]):
                errors.append(f"{prefix}.{field} 必须是 HTTPS URL")

        local_path = report["local_path"]
        if not _is_safe_local_pdf_path(local_path):
            errors.append(f"{prefix}.local_path 必须位于 data/raw/<vendor>/ 下且使用 .pdf")
        elif local_path in seen_paths:
            errors.append(f"{prefix}.local_path 重复：{local_path}")
        else:
            seen_paths.add(local_path)

        if report["format"] != "pdf":
            errors.append(f"{prefix}.format 当前阶段只允许 pdf")
        if not isinstance(report["file_size"], int) or report["file_size"] <= 0:
            errors.append(f"{prefix}.file_size 必须是正整数")
        if not isinstance(report["sha256"], str) or not SHA256_PATTERN.fullmatch(report["sha256"]):
            errors.append(f"{prefix}.sha256 必须是 64 位小写十六进制字符串")

        tags = report["tags"]
        if not isinstance(tags, list) or not tags:
            errors.append(f"{prefix}.tags 必须是非空数组")
        elif len(tags) != len(set(tags)):
            errors.append(f"{prefix}.tags 不能重复")
        elif any(not isinstance(tag, str) or not TAG_PATTERN.fullmatch(tag) for tag in tags):
            errors.append(f"{prefix}.tags 必须使用小写 kebab-case")

    if errors:
        raise ManifestValidationError("manifest 校验失败：\n- " + "\n- ".join(errors))
    return data


def load_manifest(path: Path, *, minimum_reports: int = 15) -> dict[str, Any]:
    """从 UTF-8 JSON 文件加载并校验 manifest。"""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestValidationError(f"manifest 不存在：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(f"manifest 不是有效 JSON：{exc}") from exc
    return validate_manifest(data, minimum_reports=minimum_reports)
