"""聊天展示层：通过原工厂/answer 接口使用强制 LangChain RAG 编排。"""

import hashlib
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import streamlit as st

from apt_rag.generation.providers import ProviderError, fingerprint
from apt_rag.rag.runtime import build_pipeline, load_settings

ROOT = Path(__file__).resolve().parents[1]
RAG_CONFIG = ROOT / "configs/rag.json"


def release_pipeline(resource):
    pipeline, _ = resource
    if pipeline.llm.client is not None:
        pipeline.llm.client.close()


@st.cache_resource(scope="session", max_entries=1, show_spinner=False, on_release=release_pipeline)
def get_pipeline(provider_id, allow_cloud, configuration_id, resource_owner):
    # configuration_id 纳入缓存键；Pipeline 的 fingerprint/client 不跨浏览器共享。
    return build_pipeline(ROOT, RAG_CONFIG, provider_id, allow_cloud=allow_cloud)


def revoke_consent():
    st.session_state.cloud_consent = False


def change_provider():
    revoke_consent()
    get_pipeline.clear()


def release_resources():
    revoke_consent()
    get_pipeline.clear()


def new_chat():
    chat_id = uuid4().hex
    st.session_state.chats[chat_id] = {"title": "新对话", "messages": []}
    st.session_state.active_chat = chat_id
    revoke_consent()


def initialize_state():
    if "chats" not in st.session_state:
        st.session_state.chats = {}
        st.session_state.pending = None
        st.session_state.resource_owner = uuid4().hex
        new_chat()


def error_message(error):
    # 不展示 traceback、请求对象、密钥或未经后端验证的生成原文。
    if isinstance(error, (ProviderError, ValueError)):
        return f"本次未获得通过验证的回答：{error}"
    if isinstance(error, OSError):
        return "本地文件或服务不可用。请按运行文档检查 Chunk、向量、索引、模型缓存及 Ollama。"
    return "本次处理失败，未自动重试或切换模型。请检查本地运行环境后重新提交。"


def render_message(message):
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🔎"):
        if message.get("error"):
            st.error(message["content"])
            st.caption("未自动重试；请求已发送时是否计费，以提供方账单为准。")
            return
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.caption(message["details"])
            citations = message["citations"]
            if citations:
                with st.expander(f"参考来源 · {len(citations)}", expanded=True):
                    for source in citations:
                        st.markdown(f"**[{source['label']}] {source['document_title']}**")
                        st.caption(f"{source['vendor']} · PDF 物理页码 {source['page']} · "
                                   f"章节：{source['section'] or '未标注'}")
                        st.code(source["chunk_id"], language=None)
                        st.caption(source["source_url"])
                        st.link_button("查看原报告 ↗", source["source_url"])
            else:
                st.caption("本次无引用；后端已按证据不足规则返回。")


def assistant_message(result, controls):
    # 只保留已验证答案和真实 Citation，不保存完整 Context/请求头/报告全文。
    provider = controls["provider_spec"]
    generation = result.get("generation")
    details = provider["model"]
    if generation:
        details += (f" · 生成 {generation['latency_seconds']:.1f} s"
                    f" · 输入/输出 {generation['prompt_tokens']}/{generation['output_tokens']} tokens")
    if "total_seconds" in result:
        details += f" · 管线 {result['total_seconds']:.1f} s（不含首次装载）"
    return {"role": "assistant", "content": result["answer"], "citations": result["citations"],
            "details": details, "provider_id": provider["id"]}


def process_pending(config, configuration_id, cloud_id):
    pending = st.session_state.pending
    if pending is None:
        return
    messages = st.session_state.chats[pending["chat_id"]]["messages"]
    if pending["status"] != "queued":
        # 页面重跑中断后不自动重发，避免隐式重复付费。
        messages.append({"role": "assistant", "error": True,
                         "content": "上一请求被页面重跑中断，结果与计费状态未确认；不会自动重发，请重新提交。"})
        st.session_state.pending = None
        st.rerun()
    pending["status"] = "running"
    with st.chat_message("assistant", avatar="🔎"):
        with st.spinner("正在准备模型、检索报告并生成回答… 首次装载和 CPU 重排可能需要一分钟以上。", show_time=True):
            try:
                # 每次调用都先检查授权，缓存命中不能绕过此门槛。
                allow_cloud = bool(st.session_state.cloud_consent)
                if pending["provider_id"] == cloud_id and not allow_cloud:
                    raise ValueError("DeepSeek 未获显式云端授权，未发送问题或报告片段")
                if not pending["question"].strip() or len(pending["question"]) > config["max_question_characters"]:
                    raise ValueError("问题为空或超过配置长度上限")
                pipeline, controls = get_pipeline(pending["provider_id"], allow_cloud, configuration_id,
                                                  st.session_state.resource_owner)
                result = pipeline.answer(pending["question"])
                response = assistant_message(result, controls)
            except Exception as error:
                response = {"role": "assistant", "error": True, "content": error_message(error)}
            # 先记录结果并清除队列，再更新界面；重跑不会重复生成。
            messages.append(response)
            st.session_state.pending = None
    st.rerun()


def main():
    st.set_page_config(page_title="APT 情报问答", page_icon="🔎", layout="centered", initial_sidebar_state="expanded")
    initialize_state()
    try:
        config, decision, sources = load_settings(ROOT, RAG_CONFIG)
        runtime_profile = ROOT / "configs/ollama_runtime.json"
        configuration_id = fingerprint({"config": config, "decision": decision,
            "orchestration": {"framework": "langchain", "version": version("langchain-core")},
            "ollama_runtime_sha256": hashlib.sha256(runtime_profile.read_bytes()).hexdigest()
                if runtime_profile.exists() else None})
    except Exception as error:
        st.title("APT 情报问答")
        st.error(error_message(error))
        st.info("请先准备 Phase 9 输入。运行说明见 docs/rag_pipeline.md 和 docs/streamlit_app.md。")
        st.stop()
    selection = decision["selection"]
    local_id, cloud_id = selection["local_llm_provider_id"], selection["primary_llm_provider_id"]
    providers = {p["id"]: p for p in sources["generation"]["providers"]}
    busy = st.session_state.pending is not None
    chat_titles = {key: chat["title"] for key, chat in st.session_state.chats.items()}
    with st.sidebar:
        st.title("APT 情报问答")
        st.caption("公开报告 · 可追溯证据")
        st.button("＋ 新建对话", key="new_chat", on_click=new_chat, disabled=busy, width="stretch")
        st.radio("临时会话", list(reversed(st.session_state.chats)), key="active_chat",
                 format_func=chat_titles.__getitem__,
                 on_change=revoke_consent, disabled=busy)
        st.divider()
        provider_id = st.selectbox("回答模型", [local_id, cloud_id], key="provider_id", disabled=busy,
                                  format_func=lambda key: "DeepSeek · 云端" if key == cloud_id else "Qwen · 本地",
                                  on_change=change_provider)
        st.caption(providers[provider_id]["model"])
        if provider_id == cloud_id:
            st.checkbox("本会话允许使用 DeepSeek 云端 API", key="cloud_consent", disabled=busy)
            st.caption("问题和公开报告片段将发送到 DeepSeek，生成可能计费。不要输入内部敏感信息；切换会话或模型会撤销授权。")
        else:
            st.caption("仅调用本机 Ollama，不自动切换到云端。")
        st.button("释放模型资源", key="release_resources", on_click=release_resources, disabled=busy, width="stretch")
        with st.expander("检索与使用说明"):
            st.caption("BGE-M3 → FAISS + BM25 / RRF → BGE Reranker → LLM")
            st.caption(f"{len(sources['manifest']['reports'])} 份公开报告 · Top-{decision['policies']['context_top_k']}")
            st.caption("会话仅保存在当前浏览器会话内，不写入文件。每轮独立检索，不向模型发送聊天历史，请写完整问题。")
            st.caption("引用页码为 PDF 物理页码。引用可追溯不代表事实已经独立核实。")
    chat = st.session_state.chats[st.session_state.active_chat]
    st.title("APT 情报问答")
    st.caption(f"{providers[provider_id]['model']} · 基于公开报告回答")
    if not chat["messages"]:
        st.subheader("有什么 APT 情报问题？")
        st.markdown("了解组织背景、追踪 TTP，或核查报告中的 IOC。每条回答的真实来源将显示在下方。")
        example = sources["benchmark"]["questions"][0]["question"]
        st.info(f"例如：{example}")
    for message in chat["messages"]:
        render_message(message)
    permitted = provider_id != cloud_id or bool(st.session_state.cloud_consent)
    if not permitted:
        st.info("请先在左侧勾选云端授权，或选择本地 Qwen。此时不会调用 DeepSeek。")
    question = st.chat_input("输入完整的 APT 情报问题…", key="question", disabled=busy or not permitted,
                             max_chars=config["max_question_characters"])
    if question:
        chat["messages"].append({"role": "user", "content": question})
        if chat["title"] == "新对话":
            chat["title"] = " ".join(question.split())[:28] or "新对话"
        st.session_state.pending = {"chat_id": st.session_state.active_chat, "provider_id": provider_id,
                                    "question": question, "status": "queued"}
        st.rerun()
    process_pending(config, configuration_id, cloud_id)


if __name__ == "__main__":
    main()
