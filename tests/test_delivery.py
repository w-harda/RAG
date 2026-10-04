import hashlib
import csv
import importlib.util
import json
from pathlib import Path
import re
import shutil

import pytest

from apt_rag.generation.providers import fingerprint

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reproduction_only_changes_results_path():
    original = json.loads((ROOT / "configs/embedding_benchmark.json").read_text(encoding="utf-8"))
    reproduction = json.loads((ROOT / "configs/reproduction_embedding.json").read_text(encoding="utf-8"))
    assert reproduction.pop("results_directory") == "data/processed/reproduction/phase4"
    original.pop("results_directory")
    assert reproduction == original


def test_phase4_saved_metrics_validate_without_models():
    assert script("validate_embedding_results").validate() == 3


def test_phase4_validator_rejects_tampered_metrics(tmp_path):
    for directory in ("configs", "data/benchmark", "results/phase4"):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    path = tmp_path / "results/phase4/bge-m3.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["metrics"]["mrr"] = .123
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="汇总指标"):
        script("validate_embedding_results").validate(tmp_path)


def test_manual_review_packet_is_blank_hidden_labelled_and_bound_to_original_answers():
    paths = list((ROOT / "results").rglob("*.json"))
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    packet, mapping = script("prepare_human_review").make_packet(ROOT)
    assert packet["completed_human_reviews"] == 0 and packet["independent_human_review"] is False
    assert len(packet["items"]) == len(mapping) == 80
    assert len({item["review_id"] for item in packet["items"]}) == 80
    for item in packet["items"]:
        assert "strategy" not in item and "score" not in item
        assert fingerprint(item["answer"]) == item["answer_sha256"]
        assert mapping[item["review_id"]]["answer_sha256"] == item["answer_sha256"]
        assert item["required_facts"] and item["shared_references"]
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_manual_review_export_refuses_existing_or_public_directory(tmp_path):
    exporter = script("prepare_human_review")
    with pytest.raises(ValueError, match="ignored"):
        exporter.export_packet(tmp_path, tmp_path / "results/review")
    target = tmp_path / "data/processed/review"
    target.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        exporter.export_packet(tmp_path, target)


def test_exported_review_tables_have_no_automatic_scores(tmp_path, monkeypatch):
    exporter = script("prepare_human_review")
    packet, mapping = exporter.make_packet(ROOT)
    monkeypatch.setattr(exporter, "make_packet", lambda root, **kwargs: (packet, mapping))
    target = tmp_path / "data/processed/review"
    assert exporter.export_packet(tmp_path, target) == 80
    with (target / "answer_review.csv").open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 80
    assert all(row["status"] == "pending" and row["supported_claims"] == ""
               and row["reviewer"] == "" for row in rows)
    with (target / "fact_coverage.csv").open(encoding="utf-8-sig", newline="") as source:
        facts = list(csv.DictReader(source))
    assert len(facts) == sum(len(item["required_facts"]) for item in packet["items"])
    assert all(row["coverage"] == "" for row in facts)


def test_final_document_relative_links_resolve():
    paths = [ROOT / "README.md", *(ROOT / "docs" / name for name in
              ("reproduction.md", "human_review.md", "results_summary.md", "final_delivery.md"))]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if "://" not in target:
                assert (path.parent / target.split("#")[0]).is_file(), (path, target)


def test_final_phase10_tables_match_original_summary():
    summary = json.loads((ROOT / "results/phase10/summary-v2.json").read_text(encoding="utf-8"))
    documents = [(ROOT / "README.md").read_text(encoding="utf-8"),
                 (ROOT / "docs/results_summary.md").read_text(encoding="utf-8")]
    for row in summary["results"]:
        values = ["N/A" if row[key] is None else f"{100 * row[key]:.2f}%" for key in
                  ("accuracy", "completeness", "citation_correctness", "hallucination_rate", "unverifiable_rate")]
        for document in documents:
            assert " | ".join(values) + f" | {row['accuracy_scored_answers']}/20 |" in document


def test_final_retrieval_and_embedding_tables_match_original_summaries():
    text = (ROOT / "docs/results_summary.md").read_text(encoding="utf-8")
    for row in json.loads((ROOT / "results/phase4/summary.json").read_text(encoding="utf-8"))["results"]:
        assert " | ".join(f"{row[key]:.4f}" for key in
                             ("recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@10")) in text
    for row in json.loads((ROOT / "results/phase7/summary.json").read_text(encoding="utf-8"))["results"]:
        assert " | ".join(f"{row['metrics'][key]:.4f}" for key in
                             ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10")) in text


def test_final_vectorstore_and_llm_tables_match_original_summaries():
    text = (ROOT / "docs/results_summary.md").read_text(encoding="utf-8")
    for row in json.loads((ROOT / "results/phase5/summary.json").read_text(encoding="utf-8"))["results"]:
        line = (f"| {'FAISS' if row['backend'] == 'faiss' else 'Chroma'} | {row['size']} | "
                f"{row['build_seconds']['mean']:.4f} | {row['unfiltered']['latency_ms']['p95']:.4f} | "
                f"{row['filtered']['latency_ms']['p95']:.4f} | {row['unfiltered']['ann_recall_at_k']:.4f} |")
        assert line in text
    for row in json.loads((ROOT / "results/phase6/summary.json").read_text(encoding="utf-8"))["results"]:
        quality = row["semantic_quality"]
        line = (f"| 20 | {row['latency_median_seconds']:.2f} | {row['latency_p95_seconds']:.2f} | "
                f"{100 * quality['required_fact_coverage']:.2f}% | "
                f"{100 * quality['semantic_citation_support']:.2f}% | "
                f"{100 * quality['has_unsupported_claim']:.2f}% |")
        assert line in text


@pytest.mark.parametrize("name", ["run_embedding_benchmark", "run_vectorstore_benchmark"])
def test_experiment_cli_accepts_independent_config(monkeypatch, name):
    module = script(name)
    monkeypatch.setattr("sys.argv", [name, "--config", "independent.json"])
    if name == "run_embedding_benchmark":
        assert module.parse_args().config == Path("independent.json")
    else:
        called = []
        monkeypatch.setattr(module, "run_benchmark", lambda root, path: called.append(path) or {"results": []})
        module.main()
        assert called == [Path("independent.json")]
