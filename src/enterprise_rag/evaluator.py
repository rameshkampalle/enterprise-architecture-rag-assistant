from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .answer import ask_question
from .config import Settings


@dataclass
class EvalRecord:
    question_id: str
    category: str
    question: str
    expected_sources: List[str]
    expected_facts: List[str]
    should_refuse: bool
    evaluator_notes: str = ""
    evaluation_batch: int = 1


def _normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def _is_refusal(text: str) -> bool:
    t = _normalize_text(text)
    return any(
        m in t
        for m in [
            "couldn't find",
            "could not find",
            "not found in the provided",
            "unable to answer",
            "not available in the provided",
            "i couldn't find this in the provided enterprise architecture documents",
        ]
    )


def _tokenize(text: str) -> set[str]:
    toks = _normalize_text(text).split()
    stopwords = {
        "the", "and", "for", "are", "is", "was", "were", "has", "have", "had",
        "will", "that", "this", "from", "with", "into", "your", "you", "their",
        "there", "here", "must", "shall", "should", "could", "would", "can",
        "may", "when", "where", "how", "what", "who", "which", "about", "also",
    }
    return {t for t in toks if len(t) > 2 and t not in stopwords}


def _fact_match(fact: str, answer: str) -> float:
    if not fact:
        return 1.0

    fact_norm = _normalize_text(fact)
    ans_norm = _normalize_text(answer)

    if fact_norm in ans_norm:
        return 1.0

    f_tokens = _tokenize(fact_norm)
    a_tokens = _tokenize(ans_norm)
    if not f_tokens:
        return 1.0
    if not a_tokens:
        return 0.0

    overlap = len(f_tokens & a_tokens) / len(f_tokens)
    return 1.0 if overlap >= 0.6 else overlap


def evaluate_question(
    record: EvalRecord,
    answer_text: str,
    retrieved_sources: List[str],
    top_score: float = 0.0
) -> Dict[str, Any]:
    ans = str(answer_text or "")
    is_refusal = _is_refusal(ans)

    fact_scores = [_fact_match(f, ans) for f in record.expected_facts]
    fact_coverage = sum(fact_scores) / len(fact_scores) if fact_scores else 1.0

    if record.expected_sources:
        matched = [s for s in record.expected_sources if s in set(retrieved_sources)]
        source_recall = len(matched) / len(record.expected_sources)
    else:
        source_recall = 1.0 if is_refusal else 0.0

    refusal_correct = (is_refusal == record.should_refuse)

    if record.should_refuse:
        evaluation_score = 1.0 if refusal_correct and is_refusal else 0.0
        passed = bool(evaluation_score == 1.0)
    else:
        evaluation_score = fact_coverage * 0.7 + source_recall * 0.3
        passed = bool((fact_coverage >= 0.5 or not is_refusal) and source_recall >= 0.5 and refusal_correct)

    return {
        "question_id": record.question_id,
        "category": record.category,
        "question": record.question,
        "expected_sources": record.expected_sources,
        "expected_facts": record.expected_facts,
        "should_refuse": record.should_refuse,
        "evaluator_notes": record.evaluator_notes,
        "evaluation_batch": record.evaluation_batch,
        "actual_answer": ans,
        "retrieved_sources": retrieved_sources,
        "cited_expected_sources": [s for s in record.expected_sources if s in set(retrieved_sources)],
        "top_score": top_score,
        "retrieval_count": 0,
        "fact_checks": [{"fact": f, "coverage": score, "passed": score >= 1.0} for f, score in zip(record.expected_facts, fact_scores)],
        "fact_coverage": fact_coverage,
        "source_recall": source_recall,
        "refusal_correct": refusal_correct,
        "evaluation_score": evaluation_score,
        "passed": bool(passed),
        "matched_expected_sources": [s for s in record.expected_sources if s in set(retrieved_sources)],
        "answer_is_refusal": is_refusal,
    }


def evaluate_all(settings: Settings, questions_path: Path, ground_truth_path: Path, results_out_dir: Path) -> Dict[str, Any]:
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    gt_map = {entry["question_id"]: entry for entry in json.loads(ground_truth_path.read_text(encoding="utf-8"))}

    results: List[Dict[str, Any]] = []
    for q in questions:
        gt = gt_map[q["question_id"]]
        rec = EvalRecord(
            question_id=q["question_id"],
            category=q["category"],
            question=q["question"],
            expected_sources=gt["expected_sources"],
            expected_facts=gt["expected_facts"],
            should_refuse=gt["should_refuse"],
            evaluator_notes=gt.get("evaluator_notes", ""),
            evaluation_batch=gt.get("evaluation_batch", 1),
        )
        answered = ask_question(q["question"], settings)
        row = evaluate_question(rec, answered["answer"], answered["retrieved_sources"], answered.get("retrieval_top_score", 0.0))
        row["retrieval_count"] = answered["retrieval_count"]
        results.append(row)

    questions_completed = len(results)
    questions_passed = sum(1 for r in results if r["passed"])

    def _cat(items: List[Dict[str, Any]]) -> float:
        return round(sum(i["evaluation_score"] for i in items) / len(items), 2) if items else 1.0

    direct = [r for r in results if r["category"] == "direct_factual"]
    semantic = [r for r in results if r["category"] == "semantic_paraphrased"]
    ambiguous = [r for r in results if r["category"] == "ambiguous"]
    multi = [r for r in results if r["category"] == "multi_document"]
    unanswerable = [r for r in results if r["category"] == "unanswerable"]

    summary = {
        "questions_expected": len(questions),
        "questions_completed": questions_completed,
        "questions_passed": questions_passed,
        "questions_failed": questions_completed - questions_passed,
        "overall_score": round(questions_passed / questions_completed, 2) if questions_completed else 0.0,
        "average_fact_coverage": round(sum(r["fact_coverage"] for r in results) / questions_completed, 2) if questions_completed else 0.0,
        "average_source_recall": round(sum(r["source_recall"] for r in results) / questions_completed, 2) if questions_completed else 0.0,
        "refusal_accuracy": round(sum(1 for r in results if r["refusal_correct"]) / questions_completed, 2) if questions_completed else 0.0,
        "category_scores": {
            "direct_factual": _cat(direct),
            "semantic_paraphrased": _cat(semantic),
            "ambiguous": _cat(ambiguous),
            "multi_document": _cat(multi),
            "unanswerable": _cat(unanswerable),
        },
        "failed_question_ids": [r["question_id"] for r in results if not r["passed"]],
    }

    results_out_dir.mkdir(parents=True, exist_ok=True)
    run_output = results_out_dir / "run.json"
    payload = {"summary": summary, "results": results}
    run_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = []
    for item in results:
        flat = item.copy()
        flat["expected_sources"] = "|".join(item["expected_sources"])
        flat["expected_facts"] = "|".join(item["expected_facts"])
        flat["retrieved_sources"] = "|".join(item["retrieved_sources"])
        flat["cited_expected_sources"] = "|".join(item["cited_expected_sources"])
        flat["matched_expected_sources"] = "|".join(item["matched_expected_sources"])
        flat["fact_checks"] = json.dumps(item["fact_checks"])
        rows.append(flat)

    pd.DataFrame(rows).to_csv(results_out_dir / "results.csv", index=False)
    return {"summary": summary, "results": results, "run_output": str(run_output)}