"""校验 APT 报告 manifest。"""

from collections import Counter
from pathlib import Path

from apt_rag.corpus import ManifestValidationError, load_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "data" / "manifest" / "reports.json"


def main() -> int:
    try:
        manifest = load_manifest(MANIFEST_PATH)
    except ManifestValidationError as exc:
        print(exc)
        return 1

    reports = manifest["reports"]
    vendors = Counter(report["vendor"] for report in reports)
    groups = {report["apt_group"] for report in reports}
    print(f"manifest 有效：{len(reports)} 份报告，{len(vendors)} 个来源，{len(groups)} 个组织标签")
    for vendor, count in sorted(vendors.items()):
        print(f"- {vendor}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
