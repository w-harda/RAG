"""按 manifest 下载或验证原始 APT 报告。"""

from __future__ import annotations

import argparse
from pathlib import Path

from apt_rag.corpus import ManifestValidationError, load_manifest
from apt_rag.corpus.download import ReportFileError, download_report, verify_report_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "data" / "manifest" / "reports.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-id",
        action="append",
        default=[],
        help="只处理指定 report_id；可重复传入",
    )
    parser.add_argument("--verify-only", action="store_true", help="仅校验本地文件，不访问网络")
    parser.add_argument("--overwrite", action="store_true", help="重新下载并覆盖已存在的有效文件")
    parser.add_argument("--timeout", type=float, default=120.0, help="单个网络请求的超时秒数")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = load_manifest(DEFAULT_MANIFEST)
    except ManifestValidationError as exc:
        print(exc)
        return 1

    requested = set(args.report_id)
    known_ids = {report["report_id"] for report in manifest["reports"]}
    unknown = requested - known_ids
    if unknown:
        print(f"未知 report_id：{', '.join(sorted(unknown))}")
        return 2

    reports = [
        report
        for report in manifest["reports"]
        if not requested or report["report_id"] in requested
    ]
    failures = 0
    for report in reports:
        try:
            if args.verify_only:
                path = verify_report_file(report, REPOSITORY_ROOT)
                status = "verified"
            else:
                path, status = download_report(
                    report,
                    REPOSITORY_ROOT,
                    overwrite=args.overwrite,
                    timeout=args.timeout,
                )
            print(f"[{status}] {report['report_id']} -> {path.relative_to(REPOSITORY_ROOT)}")
        except (OSError, ReportFileError) as exc:
            failures += 1
            print(f"[failed] {report['report_id']}: {exc}")

    print(f"处理完成：成功 {len(reports) - failures}，失败 {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
