#!/usr/bin/env python3
"""Summarize phase_8 Graph-RAG corpus scaling metrics by corpus size."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


PHASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PHASE_DIR / "graph_rag_corpus_scaling_results_scored.json"
DEFAULT_OUTPUT = PHASE_DIR / "graph_rag_corpus_scaling_summary.json"
DEFAULT_CSV = PHASE_DIR / "graph_rag_corpus_scaling_summary.csv"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def numeric_values(values):
    return [float(value) for value in values if value is not None]


def maybe_mean(values):
    values = numeric_values(values)
    return mean(values) if values else None


def summarize(data: dict[str, Any]) -> list[dict[str, Any]]:
    size_keys = sorted(
        data.get("covered_videos_by_size", {}).keys(),
        key=lambda value: int(value),
    )
    rows = []
    for size_key in size_keys:
        runs = [
            question.get("covered_videos", {}).get(size_key, {})
            for question in data.get("questions", [])
            if question.get("covered_videos", {}).get(size_key, {}).get("answer")
        ]
        refusal_values = [
            1.0 if run.get("is_refusal") else 0.0
            for run in runs
            if run.get("is_refusal") is not None
        ]
        correctness_values = [run.get("correctness_score") for run in runs]
        faithfulness_values = [run.get("faithfulness") for run in runs]
        evidence_counts = [run.get("final_evidence_count") for run in runs]

        row = {
            "num_videos": int(size_key),
            "n_questions": len(data.get("questions", [])),
            "n_answers": len(runs),
            "n_refusal": len(refusal_values),
            "refusal_rate": maybe_mean(refusal_values),
            "answer_rate": 1.0 - maybe_mean(refusal_values) if refusal_values else None,
            "n_correctness": len(numeric_values(correctness_values)),
            "mean_correctness": maybe_mean(correctness_values),
            "n_faithfulness": len(numeric_values(faithfulness_values)),
            "mean_faithfulness": maybe_mean(faithfulness_values),
            "mean_final_evidence_count": maybe_mean(evidence_counts),
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize phase_8 Graph-RAG corpus scaling metrics."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.input)
    rows = summarize(data)
    save_json(args.output, rows)
    write_csv(args.csv_output, rows)

    print(
        "num_videos,n_answers,refusal_rate,mean_correctness,"
        "mean_faithfulness,mean_final_evidence_count"
    )
    for row in rows:
        print(
            ",".join(
                [
                    str(row["num_videos"]),
                    str(row["n_answers"]),
                    fmt(row["refusal_rate"]),
                    fmt(row["mean_correctness"]),
                    fmt(row["mean_faithfulness"]),
                    fmt(row["mean_final_evidence_count"]),
                ]
            )
        )
    print(f"Saved summary JSON to {args.output}")
    print(f"Saved summary CSV to {args.csv_output}")


if __name__ == "__main__":
    main()
