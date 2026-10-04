"""离线校验 Phase 8 决策；复用核心校验供 Phase 9 加载使用。"""

import argparse
import json
from pathlib import Path

from apt_rag.stack_selection import json_fingerprint, validate_selection

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/final_stack.json")
    args = parser.parse_args()
    decision = json.loads(args.config.read_text(encoding="utf-8"))
    validate_selection(ROOT, decision)
    print("Phase 8 历史选型验证通过：12 份依据的内容指纹与组件选择一致；当前 RAG 接入另见 Phase 9 校验")
    for name, value in decision["selection"].items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
