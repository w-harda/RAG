from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from apt_rag.corpus.download import ReportFileError, sha256_file, verify_report_file
from apt_rag.corpus.manifest import ManifestValidationError, load_manifest, validate_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "data" / "manifest" / "reports.json"


def test_project_manifest_is_valid() -> None:
    manifest = load_manifest(MANIFEST_PATH)

    assert len(manifest["reports"]) == 15
    assert {report["vendor"] for report in manifest["reports"]} == {"CISA", "Mandiant"}


def test_duplicate_report_id_is_rejected() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    invalid = deepcopy(manifest)
    invalid["reports"].append(deepcopy(invalid["reports"][0]))

    with pytest.raises(ManifestValidationError, match="report_id 重复"):
        validate_manifest(invalid)


def test_path_traversal_is_rejected() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    invalid = deepcopy(manifest)
    invalid["reports"][0]["local_path"] = "../outside.pdf"

    with pytest.raises(ManifestValidationError, match="local_path"):
        validate_manifest(invalid)


def test_unknown_report_field_is_rejected() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    invalid = deepcopy(manifest)
    invalid["reports"][0]["unexpected"] = "value"

    with pytest.raises(ManifestValidationError, match="未知字段"):
        validate_manifest(invalid)


def test_local_pdf_integrity_check(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7\nminimal test fixture\n")
    report = {
        "report_id": "sample-report",
        "local_path": str(pdf.relative_to(tmp_path)),
        "file_size": pdf.stat().st_size,
        "sha256": sha256_file(pdf),
    }

    assert verify_report_file(report, tmp_path) == pdf

    report["sha256"] = "0" * 64
    with pytest.raises(ReportFileError, match="SHA-256"):
        verify_report_file(report, tmp_path)
