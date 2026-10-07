#!/usr/bin/env python3
"""Run Graph-RAG with progressively larger video subsets.

This script is evaluation-only. It does not rebuild caches or modify the
sanitized knowledge base. It loads the normal inference service once, then
temporarily filters ``service.stores`` to simulate smaller ingested corpora.

Subset selector:
    source_record["ablations"]["graph_rag"]["is_refusal"] is False

Output schema intentionally avoids the old ``ablations`` field. Per-question
results are stored under ``covered_videos`` keyed by corpus size.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_inference.service import InferenceService
from knowledge_inference.types import EvidenceBlock


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "knowledge_system_evaluation_v2"
    / "evaluated_datasets"
    / "community_qa_dataset_final_v3.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "knowledge_system_evaluation_v2"
    / "phase_8"
    / "graph_rag_corpus_scaling_results.json"
)
DEFAULT_TEST_OUTPUT = (
    PROJECT_ROOT
    / "knowledge_system_evaluation_v2"
    / "phase_8"
    / "graph_rag_corpus_scaling_results_test.json"
)
DEFAULT_SIZES = [5, 10, 20, 30, 38]
DEFAULT_TEST_SIZES = [5, 38]
DEFAULT_TEST_LIMIT = 2


REFUSAL_MARKERS = [
    "i do not have enough grounded evidence",
    "i don't have enough grounded evidence",
    "i do not have enough evidence",
    "i don't have enough evidence",
    "i cannot answer",
    "i can't answer",
    "cannot answer this confidently",
    "can't answer this confidently",
    "not enough information",
    "not enough context",
    "provided clips do not contain",
    "available evidence does not contain",
    "available evidence supports only part",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def build_query(item: dict[str, Any]) -> str:
    return f"{item.get('question_title', '')}\n\n{item.get('question_body', '')}".strip()


def select_graph_rag_answered(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("ablations", {}).get("graph_rag", {}).get("is_refusal") is False
    ]


def detect_refusal_heuristic(answer: str) -> bool:
    lowered = " ".join(str(answer or "").lower().split())
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def evidence_sources(evidence: list[EvidenceBlock]) -> list[str]:
    return [block.source for block in evidence]


def evidence_contexts(evidence: list[EvidenceBlock]) -> list[str]:
    return [block.text for block in evidence if isinstance(block.text, str) and block.text.strip()]


def make_question_record(source_record: dict[str, Any]) -> dict[str, Any]:
    keys_to_copy = [
        "question_id",
        "source",
        "question_url",
        "question_title",
        "question_body",
        "answer_gold",
        "temporal_status",
        "answerability_status",
        "answerability_score",
        "answerability_details",
        "tags",
        "champion_matches",
    ]
    result = {key: deepcopy(source_record.get(key)) for key in keys_to_copy if key in source_record}
    result["covered_videos"] = {}
    return result


def index_questions(existing: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in existing:
        question_id = str(item.get("question_id", "")).strip()
        if question_id:
            indexed[question_id] = item
    return indexed


def build_video_order(video_names: list[str], seed: int) -> list[str]:
    ordered = sorted(video_names)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered


def initialize_output(
    *,
    source_dataset: Path,
    output_path: Path,
    records: list[dict[str, Any]],
    subset_records: list[dict[str, Any]],
    sizes: list[int],
    seed: int,
    video_order: list[str],
    resume: bool,
    test_run: bool,
) -> dict[str, Any]:
    if resume and output_path.exists():
        output = load_json(output_path)
        existing_by_id = index_questions(output.get("questions", []))
        merged_questions = []
        for record in subset_records:
            question_id = str(record.get("question_id", "")).strip()
            merged_questions.append(existing_by_id.get(question_id) or make_question_record(record))
        output["questions"] = merged_questions
    else:
        output = {
            "experiment_metadata": {},
            "covered_videos_by_size": {},
            "questions": [make_question_record(record) for record in subset_records],
        }

    output["experiment_metadata"].update(
        {
            "schema_version": 1,
            "created_or_updated_utc": datetime.now(timezone.utc).isoformat(),
            "test_run": test_run,
            "source_dataset": str(source_dataset),
            "subset_selector": "ablations.graph_rag.is_refusal == false",
            "num_source_questions": len(records),
            "num_selected_questions": len(subset_records),
            "corpus_sizes": sizes,
            "seed": seed,
            "video_order": video_order,
            "notes": (
                "Graph-RAG corpus scaling experiment. Results are keyed by corpus "
                "size under each question's covered_videos field."
            ),
        }
    )
    output["covered_videos_by_size"] = {
        str(size): video_order[:size]
        for size in sizes
    }
    return output


async def run_scaling(args: argparse.Namespace) -> None:
    source_records = load_json(args.input)
    subset_records = select_graph_rag_answered(source_records)
    if args.limit is not None:
        subset_records = subset_records[: args.limit]

    service = InferenceService()
    service.initialize()
    all_stores = dict(service.stores)
    all_video_names = sorted(all_stores)

    if not all_video_names:
        raise RuntimeError("Inference service loaded zero video stores.")
    if max(args.sizes) > len(all_video_names):
        raise ValueError(
            f"Requested corpus size {max(args.sizes)} but only {len(all_video_names)} videos are loaded."
        )

    video_order = build_video_order(all_video_names, seed=args.seed)
    output = initialize_output(
        source_dataset=args.input,
        output_path=args.output,
        records=source_records,
        subset_records=subset_records,
        sizes=args.sizes,
        seed=args.seed,
        video_order=video_order,
        resume=args.resume,
        test_run=args.test,
    )
    save_json(args.output, output)

    questions_by_id = index_questions(output["questions"])
    total_runs = len(subset_records) * len(args.sizes)
    completed = 0

    for size in args.sizes:
        size_key = str(size)
        selected_video_names = video_order[:size]
        service.stores = {
            name: all_stores[name]
            for name in selected_video_names
        }

        for source_record in subset_records:
            question_id = str(source_record.get("question_id", "")).strip()
            if not question_id:
                raise ValueError(f"Missing question_id in record: {source_record}")
            question_record = questions_by_id[question_id]
            runs = question_record.setdefault("covered_videos", {})
            existing_run = runs.get(size_key, {})

            if args.resume and existing_run.get("answer"):
                completed += 1
                continue

            query = build_query(source_record)
            result = await service._answer_async(query=query, debug=args.debug)
            answer = result.answer
            run_payload = {
                "num_videos": size,
                "selected_videos": selected_video_names,
                "answer": answer,
                "is_refusal": detect_refusal_heuristic(answer),
                "refusal_method": "heuristic",
                "confidence": result.confidence,
                "final_evidence_count": len(result.evidence),
                "evidence_sources": evidence_sources(result.evidence),
                "evidence_contexts": evidence_contexts(result.evidence),
                "correctness_score": None,
                "correctness_reason": None,
                "faithfulness": None,
                "faithfulness_error": None,
            }
            if args.debug:
                run_payload["debug"] = result.debug

            runs[size_key] = run_payload
            completed += 1

            if completed % args.checkpoint_every == 0:
                output["experiment_metadata"]["created_or_updated_utc"] = datetime.now(timezone.utc).isoformat()
                save_json(args.output, output)
                print(f"Checkpoint saved after {completed}/{total_runs} runs")

    service.stores = all_stores
    output["experiment_metadata"]["created_or_updated_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(args.output, output)
    print(f"Saved scaling inference results to {args.output}")
    print(f"Selected questions: {len(subset_records)} / {len(source_records)}")
    print(f"Corpus sizes: {args.sizes}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Graph-RAG corpus-size scaling on the Graph-RAG answered subset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Run a small safety check. Defaults to 2 questions, corpus sizes "
            "5 and 38, checkpoint every run, and a separate *_test.json output."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional small debugging limit.")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--debug", action="store_true", help="Store full inference debug payloads.")
    args = parser.parse_args()
    if args.test:
        if args.output == DEFAULT_OUTPUT:
            args.output = DEFAULT_TEST_OUTPUT
        if args.limit is None:
            args.limit = DEFAULT_TEST_LIMIT
        if args.sizes == DEFAULT_SIZES:
            args.sizes = DEFAULT_TEST_SIZES
        args.checkpoint_every = 1
    args.sizes = sorted(dict.fromkeys(args.sizes))
    return args


def main() -> None:
    args = parse_args()
    asyncio.run(run_scaling(args))


if __name__ == "__main__":
    main()
