"""导出空白人工复核工作包；不调用 AI、不填分、不修改历史结果。"""

import argparse
import csv
import hashlib
from pathlib import Path

from apt_rag.evaluation.ablation_benchmark import load_results, read
from apt_rag.evaluation.embedding_benchmark import _write_json, corpus_fingerprint, load_chunks

ROOT = Path(__file__).resolve().parents[1]


def make_packet(root, *, include_text=False):
    _, summary = load_results(root, root / "configs/ablation_benchmark.json",
                              review_config_path=root / "configs/ablation_review_v2.json")
    answers = read(root / "results/phase10/answers.json")
    frozen = read(root / "results/phase10/inputs_manifest.json")
    benchmark = read(root / "data/benchmark/apt_qa_v1.json")
    rubric = read(root / "data/benchmark/llm_rubric_v1.json")
    reports = {r["report_id"]: r for r in read(root / "data/manifest/reports.json")["reports"]}
    questions = {q["id"]: q for q in benchmark["questions"]}
    chunks = {}
    if include_text:
        corpus = load_chunks(root / "data/processed/chunks")
        if corpus_fingerprint(corpus) != summary["controls"]["retrieval"]["corpus_fingerprint"]:
            raise ValueError("人工复核正文与冻结 Corpus 不一致")
        chunks = {c["chunk_id"]: c for c in corpus}

    def source_record(source):
        record = {**source, "download_url": reports[source["report_id"]]["download_url"]}
        if include_text:
            chunk = chunks[source["chunk_id"]]
            if any(source[key] != chunk[key] for key in
                   ("report_id", "document_title", "vendor", "page", "section", "source_url", "text_sha256")):
                raise ValueError("人工复核来源 Metadata/正文不一致")
            record["text"] = chunk["text"]
        return record

    # 隐藏策略名并确定性打散；不是严格双盲，Context 数量仍可能透露组别。
    ordered = sorted(answers["answers"], key=lambda a: hashlib.sha256(
        ("phase12-review-v1:" + a["id"]).encode()).hexdigest())
    items, mapping = [], {}
    for index, answer in enumerate(ordered, 1):
        review_id = f"H{index:03d}"
        question = questions[answer["question_id"]]
        items.append({"review_id": review_id, "question_id": question["id"],
                      "question": question["question"], "category": question["category"],
                      "expected_answer": question["expected_answer"],
                      "required_facts": rubric["required_facts"][question["id"]],
                      "ground_truth_evidence": question["evidence"], "answer": answer["answer"],
                      "answer_sha256": answer["answer_sha256"], "finish_reason": answer["finish_reason"],
                      "provided_sources": [source_record(s) for s in answer["sources"]],
                      "shared_references": [source_record(s) for s in frozen["references"][question["id"]]]})
        mapping[review_id] = {"answer_id": answer["id"], "strategy": answer["strategy"],
                              "answer_sha256": answer["answer_sha256"]}
    return {"status": "pending_independent_human_review", "independent_human_review": False,
            "completed_human_reviews": 0, "answers_fingerprint": summary["answers_fingerprint"],
            "inputs_fingerprint": summary["inputs_fingerprint"], "include_text": include_text,
            "items": items}, mapping


def csv_table(path, fields, rows):
    # UTF-8 BOM 方便常见表格工具；答案保留在 JSON，避免 CSV 公式注入/多行阅读问题。
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_packet(root, output, *, include_text=False):
    output = output.resolve()
    if not output.is_relative_to((root / "data/processed").resolve()):
        raise ValueError("人工工作包必须位于 ignored data/processed 内")
    if output.exists():
        raise FileExistsError("工作包目录已存在；使用新路径，避免覆盖人工填写内容")
    packet, mapping = make_packet(root, include_text=include_text)
    output.mkdir(parents=True)
    _write_json(output / "review_packet.json", packet)
    _write_json(output / "coordinator_mapping.json", mapping)
    csv_table(output / "answer_review.csv",
              ["review_id", "status", "reviewer", "reviewed_at", "supported_claims", "contradicted_claims",
               "unverifiable_claims", "supported_citations", "citation_occurrences", "notes"],
              [{"review_id": item["review_id"], "status": "pending"} for item in packet["items"]])
    csv_table(output / "fact_coverage.csv",
              ["review_id", "fact_index", "required_fact", "coverage", "evidence", "reason", "reviewer"],
              [{"review_id": item["review_id"], "fact_index": i, "required_fact": fact}
               for item in packet["items"] for i, fact in enumerate(item["required_facts"], 1)])
    csv_table(output / "claims.csv",
              ["review_id", "claim_id", "claim_text", "judgment", "evidence", "reason", "reviewer"], [])
    csv_table(output / "citations.csv",
              ["review_id", "occurrence_index", "source_label", "claim_text", "judgment", "evidence",
               "reason", "reviewer"], [])
    return len(packet["items"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/human-review-v1")
    parser.add_argument("--include-text", action="store_true", help="核验本地 Chunk 后加入来源和共享参考全文；不提交 Git")
    args = parser.parse_args()
    print(f"已导出 {export_packet(ROOT, args.output, include_text=args.include_text)} 个待人工复核回答；评分全部留空，未完成人工复核")


if __name__ == "__main__":
    main()
