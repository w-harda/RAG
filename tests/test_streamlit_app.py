from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from apt_rag.rag.pipeline import RAGResponseError

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app/streamlit_app.py"


@pytest.fixture
def backend(monkeypatch):
    built, answered, closed = [], [], []
    source = {"label": "S1", "document_title": "测试报告标题", "vendor": "CISA", "page": 7,
              "section": "组织背景", "source_url": "https://www.cisa.gov/test-report",
              "report_id": "test-report", "chunk_id": "test-report-p0007-c000", "text_sha256": "test"}

    def factory(root, config, provider_id, *, allow_cloud=False):
        built.append((provider_id, allow_cloud))
        spec = {"id": provider_id, "model": "deepseek-flash" if provider_id == "deepseek-flash" else "qwen3.5:4b"}

        def answer(question):
            answered.append((provider_id, question))
            return {"answer": "**测试回答**：真实字段由后端返回 [S1]。", "citations": [deepcopy(source)],
                    "generation": {"latency_seconds": .5, "prompt_tokens": 100, "output_tokens": 20},
                    "total_seconds": 1.5, "messages": [{"content": "不应存入 UI 的完整 Context"}]}

        pipeline = SimpleNamespace(answer=answer, llm=SimpleNamespace(
            client=SimpleNamespace(close=lambda: closed.append(provider_id))))
        return pipeline, {"provider_spec": spec}

    monkeypatch.setattr("apt_rag.rag.runtime.build_pipeline", factory)
    return SimpleNamespace(built=built, answered=answered, closed=closed, source=source)


def app():
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    return at


def test_start_is_lazy_local_default_no_models_or_api(backend, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    at = app()
    assert at.selectbox[0].value == "ollama-qwen"
    assert not at.chat_input[0].disabled and not at.checkbox
    assert backend.built == backend.answered == []
    assert len(at.radio[0].options) == 1


def test_local_chat_markdown_citation_fields_and_no_duplicate_on_rerun(backend):
    at = app()
    at.chat_input[0].set_value("APT44 的组织背景？").run()
    assert not at.exception and len(at.chat_message) == 2
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assert any("**测试回答**" in m.value for m in at.markdown)
    assert any("测试报告标题" in m.value for m in at.markdown)
    assert any("物理页码 7" in c.value and "组织背景" in c.value for c in at.caption)
    assert any(backend.source["source_url"] in c.value for c in at.caption)
    assert at.code[0].value == backend.source["chunk_id"]
    assert at.get("link_button")[0].proto.url == backend.source["source_url"]
    history = at.session_state.chats[at.session_state.active_chat]["messages"]
    assert "messages" not in history[-1]
    assert "完整 Context" not in str(history)
    assert backend.built == [("ollama-qwen", False)]
    assert backend.answered == [("ollama-qwen", "APT44 的组织背景？")]
    at.run()
    assert len(backend.answered) == 1
    at.chat_input[0].set_value("Volt Typhoon 的 TTP？").run()
    assert len(backend.built) == 1 and len(backend.answered) == 2


def test_cloud_explicit_consent_and_cached_pipeline_cannot_bypass_revocation(backend):
    at = app()
    at.selectbox[0].select("deepseek-flash").run()
    assert at.chat_input[0].disabled and not at.checkbox[0].value
    assert backend.built == []
    at.checkbox[0].check().run()
    at.chat_input[0].set_value("APT44 是什么？").run()
    assert not at.exception
    assert backend.built == [("deepseek-flash", True)]
    assert backend.answered == [("deepseek-flash", "APT44 是什么？")]
    at.checkbox[0].uncheck().run()
    assert at.chat_input[0].disabled
    # 即使队列已经存在，服务调用之前仍有独立授权检查（包括缓存命中）。
    at.session_state.pending = {"chat_id": at.session_state.active_chat, "provider_id": "deepseek-flash",
                                "question": "不应发送的问题", "status": "queued"}
    at.run()
    assert len(backend.answered) == 1 and len(backend.built) == 1
    assert any("未获显式云端授权" in e.value for e in at.error)


def test_runtime_profile_change_does_not_reuse_old_pipeline(backend, monkeypatch):
    original = Path.read_bytes
    revision = [b"first profile"]
    profile = ROOT / "configs/ollama_runtime.json"
    monkeypatch.setattr(Path, "read_bytes", lambda path: revision[0] if path == profile else original(path))
    at = app()
    at.chat_input[0].set_value("第一个问题").run()
    revision[0] = b"updated profile"
    at.chat_input[0].set_value("第二个问题").run()
    assert not at.exception
    assert backend.built == [("ollama-qwen", False), ("ollama-qwen", False)]
    assert len(backend.answered) == 2


def test_history_new_chat_switch_revokes_consent_without_calling_backend(backend):
    at = app()
    first = at.session_state.active_chat
    at.selectbox[0].select("deepseek-flash").run()
    at.checkbox[0].check().run()
    at.chat_input[0].set_value("APT44 的背景").run()
    at.button(key="new_chat").click().run()
    assert len(at.session_state.chats) == 2 and not at.checkbox[0].value
    assert not at.chat_message and at.chat_input[0].disabled
    at.checkbox[0].check().run()
    at.radio[0].set_value(first).run()
    assert len(at.chat_message) == 2 and not at.checkbox[0].value
    assert len(backend.answered) == 1


def test_switch_provider_releases_client_no_automatic_fallback(backend):
    at = app()
    at.chat_input[0].set_value("公开问题").run()
    at.selectbox[0].select("deepseek-flash").run()
    assert backend.closed == ["ollama-qwen"]
    assert not at.checkbox[0].value and at.chat_input[0].disabled
    assert backend.built == [("ollama-qwen", False)]


def test_browser_sessions_do_not_share_mutable_pipeline(backend):
    first, second = app(), app()
    first.chat_input[0].set_value("第一个浏览器").run()
    second.chat_input[0].set_value("第二个浏览器").run()
    assert len(backend.built) == 2
    assert len(first.session_state.chats) == len(second.session_state.chats) == 1
    assert "第二个浏览器" not in str(first.session_state.chats)


@pytest.mark.parametrize("question", [" ", "长" * 1001])
def test_invalid_question_never_loads_backend(backend, question):
    at = app()
    at.chat_input[0].set_value(question).run()
    assert backend.built == [] and backend.answered == []
    assert any("问题为空或超过" in e.value for e in at.error)


@pytest.mark.parametrize("failure", [
    RAGResponseError("回答包含未知来源标签", {"generation": {"answer": "UNVERIFIED-ANSWER-MUST-NOT-SHOW"}}),
    RuntimeError("SENSITIVE-INTERNAL-DETAIL-MUST-NOT-SHOW"),
])
def test_rejected_answer_or_unexpected_error_not_rendered_or_retried(monkeypatch, failure):
    calls = []

    def factory(*args, **kwargs):
        def answer(question):
            calls.append(question)
            raise failure
        return SimpleNamespace(answer=answer, llm=SimpleNamespace(client=None)), {"provider_spec": {"id": "local", "model": "test"}}

    monkeypatch.setattr("apt_rag.rag.runtime.build_pipeline", factory)
    at = app()
    at.chat_input[0].set_value("公开问题").run()
    assert not at.exception and at.error
    visible = str([x.value for x in at.markdown]) + str([x.value for x in at.error])
    assert "UNVERIFIED-ANSWER" not in visible and "SENSITIVE-INTERNAL" not in visible
    at.run()
    assert calls == ["公开问题"]


def test_interrupted_pending_request_is_not_automatically_resent(backend):
    at = app()
    at.session_state.pending = {"chat_id": at.session_state.active_chat, "provider_id": "ollama-qwen",
                                "question": "上一次的问题", "status": "running"}
    at.run()
    assert backend.built == backend.answered == []
    assert at.session_state.pending is None
    assert any("不会自动重发" in e.value for e in at.error)


def test_missing_input_is_safe_readable_and_never_loads_backend(backend, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("not-ready")
    monkeypatch.setattr("apt_rag.rag.runtime.load_settings", missing)
    at = app()
    assert at.error and not at.chat_input
    assert backend.built == []
