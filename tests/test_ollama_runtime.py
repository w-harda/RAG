"""新版本地运行兼容与拒绝路径；假 HTTP 不访问模型或云端。"""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from apt_rag.generation.providers import OllamaProvider, fingerprint
from apt_rag.rag.budget import TokenBudget
from apt_rag.rag.runtime import configure_ollama_runtime

ROOT = Path(__file__).resolve().parents[1]
PROFILE = json.loads((ROOT / "configs/ollama_runtime.json").read_text(encoding="utf-8"))
BASE_CONFIG = json.loads((ROOT / "configs/rag.json").read_text(encoding="utf-8"))
BASE_SPEC = json.loads((ROOT / "configs/llm_benchmark.json").read_text(encoding="utf-8"))["providers"][0]


@pytest.mark.parametrize("version", ["0.40.0", "0.40.1"])
def test_runtime_profile_preserves_baseline_and_records_actual_backend(version):
    config, spec = deepcopy(BASE_CONFIG), deepcopy(BASE_SPEC)
    provider = SimpleNamespace(host="http://127.0.0.1", spec=spec,
                               request=lambda *a: {"version": version})
    active_config, active_spec, audit = configure_ollama_runtime(ROOT, config, spec, provider)
    assert config == BASE_CONFIG and spec == BASE_SPEC
    assert active_config["ollama_version"] == active_spec["ollama_version"] == audit["ollama_version"] == version
    assert active_spec["digest"] == PROFILE["digest"] and active_spec["runner"] == "ggml"
    assert provider.spec is active_spec
    assert audit["config_fingerprint"] == fingerprint(PROFILE)
    assert audit["baseline_model_digest"] == BASE_SPEC["digest"]


def test_original_version_keeps_original_config_and_provider():
    provider = SimpleNamespace(host="http://127.0.0.1", spec=BASE_SPEC,
                               request=lambda *a: {"version": BASE_CONFIG["ollama_version"]})
    config, spec, audit = configure_ollama_runtime(ROOT, BASE_CONFIG, BASE_SPEC, provider)
    assert config is BASE_CONFIG and spec is BASE_SPEC and audit is None


@pytest.mark.parametrize("version", ["0.39.0", "0.40.2", "0.41.0"])
def test_unverified_versions_are_rejected(version):
    provider = SimpleNamespace(host="http://127.0.0.1", request=lambda *a: {"version": version})
    with pytest.raises(ValueError, match="未验证"):
        configure_ollama_runtime(ROOT, BASE_CONFIG, BASE_SPEC, provider)


def test_missing_profile_is_not_an_automatic_version_bypass(tmp_path):
    provider = SimpleNamespace(host="http://127.0.0.1", request=lambda *a: {"version": "0.40.0"})
    with pytest.raises(ValueError, match="缺少运行兼容配置"):
        configure_ollama_runtime(tmp_path, BASE_CONFIG, BASE_SPEC, provider)


@pytest.mark.parametrize("versions", [None, [], "0.40.1", [1], ["*"], ["0.40.*"], ["0.40.1", "0.40.1"]])
def test_invalid_version_list_is_not_a_wildcard_or_implicit_bypass(tmp_path, versions):
    path = tmp_path / "configs/ollama_runtime.json"
    path.parent.mkdir()
    path.write_text(json.dumps({**PROFILE, "verified_ollama_versions": versions}), encoding="utf-8")
    provider = SimpleNamespace(host="http://127.0.0.1", request=lambda *a: {"version": "0.40.1"})
    with pytest.raises(ValueError, match="版本列表无效"):
        configure_ollama_runtime(tmp_path, BASE_CONFIG, BASE_SPEC, provider)


@pytest.mark.parametrize("field,value", [("baseline_model_digest", "0" * 64),
    ("runner", "llamacpp"), ("digest", "invalid"), ("template_sha256", "invalid")])
def test_tampered_profile_is_rejected(tmp_path, field, value):
    path = tmp_path / "configs/ollama_runtime.json"
    path.parent.mkdir()
    path.write_text(json.dumps({**PROFILE, field: value}), encoding="utf-8")
    provider = SimpleNamespace(host="http://127.0.0.1", request=lambda *a: {"version": "0.40.0"})
    with pytest.raises(ValueError, match="基线不一致|指纹无效"):
        configure_ollama_runtime(tmp_path, BASE_CONFIG, BASE_SPEC, provider)


def provider_fixture(monkeypatch, *, digest=None, runner="ggml", template="test", rendered="s q", token="a", version="0.40.0", pinned_version="0.40.0"):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    requests = []
    spec = {**BASE_SPEC, "digest": PROFILE["digest"], "runner": "ggml",
            "template_sha256": fingerprint("test"), "ollama_version": pinned_version}

    def handle(request):
        body = json.loads(request.content) if request.content else None
        requests.append((request.url.path, body))
        if request.url.path == "/api/version":
            data = {"version": version}
        elif request.url.path == "/api/tags":
            data = {"models": [
                {"name": spec["model"], "digest": "wrong", "size": 1, "details": {"runner": "llamacpp"}},
                {"name": spec["model"], "digest": digest or spec["digest"], "size": 1, "details": {"runner": "ggml"}},
            ]}
        elif request.url.path == "/api/show":
            data = {"capabilities": ["thinking"], "details": {"runner": runner}, "template": template,
                    "model_info": {"tokenizer.ggml.pre": "qwen35", "tokenizer.ggml.tokens": [token],
                                   "tokenizer.ggml.merges": []}}
        elif body.get("_debug_render_only"):
            data = {"_debug_info": {"rendered_template": rendered}} if rendered is not None else {}
        else:
            data = {"done": True, "done_reason": "stop", "model": spec["model"],
                    "message": {"content": "test"}, "prompt_eval_count": 3, "eval_count": 2}
        return httpx.Response(200, json=data)

    client = httpx.Client(transport=httpx.MockTransport(handle))
    settings = {"timeout_seconds": 10, "temperature": 0, "max_output_tokens": 5, "strict_context": True}
    return OllamaProvider(spec, settings, client=client), requests


def counter_fixture(provider):
    counter = TokenBudget.__new__(TokenBudget)
    counter.provider, counter.spec, counter.native_verified = provider, provider.spec, False
    counter.config = {**BASE_CONFIG, "ollama_version": provider.spec["ollama_version"]}
    counter.output_tokens, counter.token_spec = 5, {"tokenizer_sha256": "test"}
    counter.backend_json = {"model": {"merges": []}}
    counter.tokenizer = SimpleNamespace(get_vocab=lambda: {"a": 0},
        encode=lambda *a, **k: SimpleNamespace(ids=[0] * 3))
    return counter


@pytest.mark.parametrize("version", ["0.40.0", "0.40.1"])
def test_duplicate_tags_and_native_precheck_use_same_pinned_runner(monkeypatch, version):
    provider, requests = provider_fixture(monkeypatch, version=version, pinned_version=version)
    try:
        identity = provider.identity()
        assert identity["digest"] == PROFILE["digest"] and identity["details"]["runner"] == "ggml"
        counter = counter_fixture(provider)
        assert counter.measure([{"role": "system", "content": "s"},
                                {"role": "user", "content": "q"}])["input_tokens"] == 3
        body = next(body for path, body in requests if path == "/api/chat")
        assert body["runner"] == "ggml" and body["_debug_render_only"] is True
        assert body["truncate"] is False and body["shift"] is False
        assert all(body["runner"] == "ggml" for path, body in requests if path == "/api/show")
        result = provider.generate([])
        counter.verify_usage(result, {"input_tokens": 3, "safety_margin": 256, "context_window_tokens": 8192})
        generation_body = requests[-1][1]
        assert generation_body["runner"] == "ggml" and "_debug_render_only" not in generation_body
        assert generation_body["truncate"] is False and generation_body["shift"] is False
    finally:
        provider.client.close()


@pytest.mark.parametrize("options,match", [({"rendered": None}, "完整模板"),
    ({"rendered": "s"}, "完整模板"), ({"token": "changed"}, "tokenizer")])
def test_runtime_compatibility_does_not_disable_native_prechecks(monkeypatch, options, match):
    provider, requests = provider_fixture(monkeypatch, **options)
    try:
        with pytest.raises(ValueError, match=match):
            counter_fixture(provider).measure([{"role": "system", "content": "s"},
                                               {"role": "user", "content": "q"}])
        assert not any(path == "/api/chat" and not body.get("_debug_render_only") for path, body in requests)
    finally:
        provider.client.close()


@pytest.mark.parametrize("options,match", [({"digest": "wrong"}, "digest"),
    ({"runner": "llamacpp"}, "runner"), ({"template": "changed"}, "模板指纹"),
    ({"version": "0.41.0"}, "运行版本")])
def test_model_runner_and_template_changes_are_rejected(monkeypatch, options, match):
    provider, _ = provider_fixture(monkeypatch, **options)
    try:
        with pytest.raises(ValueError, match=match):
            provider.identity()
    finally:
        provider.client.close()
