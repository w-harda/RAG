"""不截断 Chunk 的 Context 构造和来源标签校验，不判断断言的语义正确性。"""

import hashlib
import re
from typing import Any

from apt_rag.generation.providers import fingerprint

SOURCE_FIELDS = ("chunk_id", "report_id", "document_title", "vendor", "page", "section", "source_url", "text_sha256")
ABSTENTION = "证据不足，无法依据当前报告回答。"


class CitationError(ValueError):
    pass


def construct_context(question: str, chunks: list[dict], prompt: dict) -> dict:
    sources, blocks = [], []
    if len({c["chunk_id"] for c in chunks}) != len(chunks):
        raise ValueError("Context 不能有重复 Chunk")
    for index, chunk in enumerate(chunks, start=1):
        if hashlib.sha256(chunk["text"].encode()).hexdigest() != chunk["text_sha256"]:
            raise ValueError("Chunk 正文与指纹不一致")
        source = {key: chunk[key] for key in SOURCE_FIELDS}
        source["label"] = f"S{index}"
        sources.append(source)
        blocks.append(
            f"[{source['label']}] {source['document_title']} | {source['vendor']} | "
            f"PDF 物理页码 {source['page']} | Chunk {source['chunk_id']}\n"
            f"URL: {source['source_url']}\n{chunk['text']}"
        )
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user_template"].format(
            question=question, context="\n\n".join(blocks),
        )},
    ]
    return {"sources": sources, "messages": messages, "messages_sha256": fingerprint(messages)}


def resolve_citations(answer: str, sources: list[dict[str, Any]]) -> dict:
    by_label = {s["label"]: s for s in sources}
    occurrences = []
    if re.search(r"\[\s*S\d+[^\]\n]*(?:\n|$)", answer, re.IGNORECASE):
        raise CitationError("回答包含未闭合来源标签")
    for group in re.findall(r"\[([^\]\n]*)\]", answer):
        # S<number> 保留为来源命名空间；畸形组合不能部分通过。
        if not re.search(r"\bS\d+", group, re.IGNORECASE):
            continue
        if not re.fullmatch(r"S[1-9]\d*(?:\s*[,，]\s*S[1-9]\d*)*", group):
            raise CitationError("回答包含畸形来源标签")
        occurrences.extend(re.findall(r"S[1-9]\d*", group))
    if set(occurrences) - by_label.keys():
        raise CitationError("回答包含未知来源标签；禁止编造引用")
    if not occurrences:
        if answer.strip() != ABSTENTION:
            raise CitationError("回答未包含来源标签，不能作为有证据的成功回答")
        return {"occurrences": [], "citations": [], "abstained": True}
    unique = list(dict.fromkeys(occurrences))
    return {"occurrences": occurrences, "citations": [by_label[label] for label in unique], "abstained": False}
