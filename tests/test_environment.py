import sys

import apt_rag


def test_python_version_is_312() -> None:
    assert sys.version_info[:2] == (3, 12)


def test_package_is_importable() -> None:
    assert apt_rag.__version__ == "0.1.0"
