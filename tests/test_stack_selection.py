"""Phase 8 决策仅依赖 Git 中配置/结果，不需要密钥、模型或处理后语料。"""

import json
import runpy
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
validator = runpy.run_path(str(ROOT / "scripts/validate_stack_selection.py"))


def decision():
    return json.loads((ROOT / "configs/final_stack.json").read_text(encoding="utf-8"))


def test_committed_selection_has_complete_frozen_evidence():
    sources = validator["validate_selection"](ROOT, decision())
    assert len(sources) == 12
    assert sources["phase7"]["scored_pairs"] == 600


@pytest.mark.parametrize("field,value", [
    ("embedding_model_id", "unmeasured"), ("vectorstore_backend", "milvus"),
    ("retrieval_strategy", "dense_reranker"), ("primary_llm_provider_id", "unmeasured"),
    ("local_llm_provider_id", "deepseek-flash"),
])
def test_unmeasured_or_nonlocal_component_is_rejected(field, value):
    record = decision()
    record["selection"][field] = value
    with pytest.raises(ValueError):
        validator["validate_selection"](ROOT, record)


def test_changed_evidence_and_missing_reference_are_rejected():
    record = decision()
    record["references"]["phase7"]["json_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="内容指纹"):
        validator["validate_selection"](ROOT, record)
    del record["references"]["phase7"]
    with pytest.raises(ValueError, match="完整覆盖"):
        validator["validate_selection"](ROOT, record)


def test_content_fingerprint_is_independent_of_formatting():
    value = {"a": "中文", "b": [1, 2]}
    reformatted = json.loads(json.dumps(value, indent=4).replace("\n", "\r\n"))
    assert validator["json_fingerprint"](value) == validator["json_fingerprint"](reformatted)


def test_no_automatic_cloud_fallback_or_integration_claim():
    record = decision()
    record["policies"]["automatic_provider_fallback"] = True
    with pytest.raises(ValueError, match="政策"):
        validator["validate_selection"](ROOT, record)
    record = decision()
    record["status"] = "integrated"
    with pytest.raises(ValueError, match="已集成"):
        validator["validate_selection"](ROOT, record)


def test_reference_cannot_escape_repository():
    record = deepcopy(decision())
    record["references"]["processing"]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="越出"):
        validator["validate_selection"](ROOT, record)
