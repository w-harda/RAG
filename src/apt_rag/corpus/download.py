"""下载并验证 manifest 中声明的原始报告。"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

USER_AGENT = "apt-rag-corpus/0.1 (+https://github.com/w-harda/RAG)"


class ReportFileError(RuntimeError):
    """报告下载或完整性校验失败。"""


def _download_with_urllib(url: str, destination: Path, timeout: float) -> None:
    request = Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/pdf"},
    )
    with urlopen(request, timeout=timeout) as response, destination.open("wb") as target:
        shutil.copyfileobj(response, target, length=1024 * 1024)


def _download_with_curl(url: str, destination: Path, timeout: float) -> None:
    curl = shutil.which("curl")
    if curl is None:
        raise ReportFileError("Python 下载被来源站点拒绝，且系统中未找到 curl")
    command = [
        curl,
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--max-time",
        str(timeout),
        "--user-agent",
        USER_AGENT,
        "--header",
        "Accept: application/pdf",
        "--output",
        str(destination),
        url,
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise ReportFileError(f"curl 下载失败，退出码 {exc.returncode}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_report_file(report: dict[str, Any], repository_root: Path) -> Path:
    path = repository_root / report["local_path"]
    if not path.is_file():
        raise ReportFileError(f"文件不存在：{report['local_path']}")
    if path.stat().st_size != report["file_size"]:
        raise ReportFileError(
            f"文件大小不匹配：{report['report_id']} "
            f"expected={report['file_size']} actual={path.stat().st_size}"
        )
    with path.open("rb") as source:
        if source.read(5) != b"%PDF-":
            raise ReportFileError(f"文件不是有效 PDF：{report['local_path']}")
    actual_hash = sha256_file(path)
    if actual_hash != report["sha256"]:
        raise ReportFileError(
            f"SHA-256 不匹配：{report['report_id']} "
            f"expected={report['sha256']} actual={actual_hash}"
        )
    return path


def download_report(
    report: dict[str, Any],
    repository_root: Path,
    *,
    overwrite: bool = False,
    timeout: float = 120.0,
) -> tuple[Path, str]:
    """下载单份报告；已存在且校验通过的文件不会重复下载。"""

    destination = repository_root / report["local_path"]
    if destination.exists() and not overwrite:
        return verify_report_file(report, repository_root), "verified"

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        try:
            _download_with_urllib(report["download_url"], partial, timeout)
        except (HTTPError, URLError):
            partial.unlink(missing_ok=True)
            _download_with_curl(report["download_url"], partial, timeout)

        temporary_report = dict(report)
        temporary_report["local_path"] = str(partial.relative_to(repository_root)).replace("\\", "/")
        verify_report_file(temporary_report, repository_root)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    return destination, "downloaded"
