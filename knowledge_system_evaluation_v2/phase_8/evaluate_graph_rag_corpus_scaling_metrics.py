#!/usr/bin/env python3
"""Evaluate correctness, faithfulness, and refusal labels for phase_8 scaling.

The input is the JSON produced by ``run_graph_rag_corpus_scaling.py``.
The script supports resume: if the output exists, it loads the output and only
evaluates missing metric fields.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
import time
import types
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "knowledge_system_evaluation_v2"
    / "phase_8"
    / "graph_rag_corpus_scaling_results.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "knowledge_system_evaluation_v2"
    / "phase_8"
    / "graph_rag_corpus_scaling_results_scored.json"
)
DEFAULT_METRICS = ["refusal", "correctness", "faithfulness"]
OPENAI_RETRY_ATTEMPTS = 5


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def build_question_text(item: dict[str, Any]) -> str:
    return f"{item.get('question_title', '')}\n\n{item.get('question_body', '')}".strip()


def get_run(item: dict[str, Any], size_key: str) -> dict[str, Any]:
    return item.setdefault("covered_videos", {}).setdefault(size_key, {})


def iter_runs(data: dict[str, Any], sizes: list[str] | None = None):
    requested = set(sizes) if sizes else None
    for q_idx, item in enumerate(data.get("questions", [])):
        question = build_question_text(item)
        answer_gold = item.get("answer_gold", "")
        for size_key, run in item.get("covered_videos", {}).items():
            if requested is not None and size_key not in requested:
                continue
            if not run.get("answer"):
                continue
            yield q_idx, size_key, item, run, question, answer_gold


def call_openai_json_with_retries(prompt: str, required_keys: list[str], model: str) -> dict[str, Any]:
    last_error = None
    for attempt in range(OPENAI_RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a strict JSON-outputting academic evaluator."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            missing = [key for key in required_keys if key not in result]
            if missing:
                raise ValueError(f"Missing required JSON keys: {missing}; content={content}")
            return result
        except Exception as exc:
            last_error = exc
            if attempt == OPENAI_RETRY_ATTEMPTS - 1:
                break
            sleep_s = min(30.0, (2 ** attempt) + random.uniform(0.0, 0.75))
            time.sleep(sleep_s)
    raise RuntimeError(f"OpenAI JSON call failed after {OPENAI_RETRY_ATTEMPTS} attempts: {last_error}")


def evaluate_correctness_task(task: tuple[int, str, str, str, str, str]):
    q_idx, size_key, question, answer_gold, generated_answer, model = task
    prompt = f"""You are an impartial expert judge evaluating an AI-generated answer for a League of Legends knowledge base.

[Inputs]
Question: {question}
Gold Answer (Community Accepted): {answer_gold}
Generated Answer (System Output): {generated_answer}

[Task]
Evaluate the Reference Answer Correctness of the Generated Answer compared to the Gold Answer.
Do NOT penalize the Generated Answer if it includes extra *correct* context, as long as it solves the problem equally well.

[Scoring Rubric - Strict 0 to 2 Scale]
0 (Incorrect or non-answer): The response is materially wrong, contradicts the reference answer, invents important facts, or does not answer the question.
1 (Partially correct): The response captures some of the relevant answer but has a material omission, ambiguity, or factual error compared to the Gold Answer.
2 (Substantially correct): The response directly and correctly answers the question and is consistent with the Gold Answer.

Respond ONLY with a JSON object in this exact format:
{{
    "correctness_score": <int 0, 1, or 2>,
    "correctness_reason": "<1-sentence justification>"
}}
"""
    try:
        result = call_openai_json_with_retries(
            prompt,
            required_keys=["correctness_score", "correctness_reason"],
            model=model,
        )
        score = result.get("correctness_score")
        if score is not None:
            score = int(score)
        return q_idx, size_key, score, result.get("correctness_reason", ""), None
    except Exception as exc:
        return q_idx, size_key, None, None, f"Error: {exc}"


def evaluate_refusal_task(task: tuple[int, str, str, str, str]):
    q_idx, size_key, question, generated_answer, model = task
    prompt = f"""You are an expert AI evaluator.
Your task is to determine whether an AI system refused to answer a user's question.

[Inputs]
Question: {question}
Generated Answer: {generated_answer}

[Instructions]
1. Set "is_refusal" to true only if the Generated Answer explicitly states that it cannot answer, cannot determine, lacks enough information, lacks context/evidence, or that the provided clips/context do not contain the answer.
2. Do not mark cautious answers as refusals if they still provide a substantive answer.
3. Provide a brief 1-sentence reason.

Respond ONLY with a JSON object in this exact format:
{{
    "is_refusal": true/false,
    "reason": "..."
}}
"""
    try:
        result = call_openai_json_with_retries(
            prompt,
            required_keys=["is_refusal", "reason"],
            model=model,
        )
        return q_idx, size_key, bool(result.get("is_refusal")), result.get("reason", ""), None
    except Exception as exc:
        return q_idx, size_key, None, None, f"Error: {exc}"


def run_refusal_judge(data: dict[str, Any], args: argparse.Namespace, sizes: list[str] | None) -> None:
    tasks = []
    for q_idx, size_key, _item, run, question, _answer_gold in iter_runs(data, sizes=sizes):
        if run.get("refusal_method") == "judge" and run.get("is_refusal") is not None:
            continue
        tasks.append((q_idx, size_key, question, run.get("answer", ""), args.openai_model))

    print(f"Pending refusal judge API calls: {len(tasks)}")
    if not tasks:
        return

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(evaluate_refusal_task, task) for task in tasks]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Refusal judge"):
            q_idx, size_key, is_refusal, reason, error = future.result()
            run = get_run(data["questions"][q_idx], size_key)
            if error:
                run["refusal_error"] = error
            else:
                run["is_refusal"] = is_refusal
                run["refusal_reason"] = reason
                run["refusal_method"] = "judge"
                run.pop("refusal_error", None)

            completed += 1
            if completed % args.checkpoint_every == 0:
                save_json(args.output, data)

    save_json(args.output, data)


def run_correctness(data: dict[str, Any], args: argparse.Namespace, sizes: list[str] | None) -> None:
    tasks = []
    for q_idx, size_key, _item, run, question, answer_gold in iter_runs(data, sizes=sizes):
        if run.get("correctness_score") is not None:
            continue
        tasks.append((q_idx, size_key, question, answer_gold, run.get("answer", ""), args.openai_model))

    print(f"Pending correctness API calls: {len(tasks)}")
    if not tasks:
        return

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(evaluate_correctness_task, task) for task in tasks]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Correctness"):
            q_idx, size_key, score, reason, error = future.result()
            run = get_run(data["questions"][q_idx], size_key)
            if error:
                run["correctness_error"] = error
            else:
                run["correctness_score"] = score
                run["correctness_reason"] = reason
                run.pop("correctness_error", None)

            completed += 1
            if completed % args.checkpoint_every == 0:
                save_json(args.output, data)

    save_json(args.output, data)


def run_faithfulness(data: dict[str, Any], args: argparse.Namespace, sizes: list[str] | None) -> None:
    # RAGAS imports are intentionally lazy. Some installed RAGAS versions import
    # optional VertexAI adapters even when OpenAI-only scoring is used. The shim
    # keeps this script usable in the current local evaluation environment.
    if "langchain_community.chat_models.vertexai" not in sys.modules:
        vertexai_module = types.ModuleType("langchain_community.chat_models.vertexai")

        class ChatVertexAI:  # pragma: no cover - compatibility shim only
            pass

        vertexai_module.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = vertexai_module

    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import llm_factory
    from ragas.metrics import Faithfulness
    from ragas.run_config import RunConfig

    ragas_payload = {"question": [], "answer": [], "contexts": [], "q_idx": [], "size_key": []}
    for q_idx, size_key, _item, run, question, _answer_gold in iter_runs(data, sizes=sizes):
        if run.get("faithfulness") is not None:
            continue
        contexts = [
            context
            for context in run.get("evidence_contexts", [])
            if isinstance(context, str) and context.strip()
        ]
        if not contexts:
            run["faithfulness"] = None
            run["faithfulness_error"] = "No retrieved evidence contexts available."
            continue
        ragas_payload["question"].append(question)
        ragas_payload["answer"].append(run.get("answer", ""))
        ragas_payload["contexts"].append(contexts)
        ragas_payload["q_idx"].append(q_idx)
        ragas_payload["size_key"].append(size_key)

    print(f"Pending faithfulness RAGAS evaluations: {len(ragas_payload['question'])}")
    if not ragas_payload["question"]:
        save_json(args.output, data)
        return

    llm_judge = llm_factory(args.openai_model)
    dataset = Dataset.from_dict(ragas_payload)
    result = evaluate(
        dataset,
        metrics=[Faithfulness(llm=llm_judge)],
        run_config=RunConfig(timeout=args.ragas_timeout, max_workers=args.ragas_max_workers),
        raise_exceptions=False,
    )

    df = result.to_pandas()
    for idx, row in df.iterrows():
        q_idx = int(dataset["q_idx"][idx])
        size_key = dataset["size_key"][idx]
        run = get_run(data["questions"][q_idx], size_key)
        faithfulness = float(row["faithfulness"]) if pd.notna(row.get("faithfulness")) else None
        if faithfulness is None:
            run["faithfulness_error"] = "RAGAS returned null/NaN faithfulness."
        else:
            run["faithfulness"] = faithfulness
            run.pop("faithfulness_error", None)

    save_json(args.output, data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score phase_8 Graph-RAG corpus scaling results."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metrics", nargs="+", choices=DEFAULT_METRICS, default=DEFAULT_METRICS)
    parser.add_argument("--sizes", nargs="+", default=None, help="Optional subset of size keys, e.g. 5 10 38.")
    parser.add_argument("--openai-model", default="gpt-5.4-nano")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--ragas-max-workers", type=int, default=3)
    parser.add_argument("--ragas-timeout", type=int, default=600)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_path = args.output if args.resume and args.output.exists() else args.input
    data = load_json(load_path)
    save_json(args.output, data)
    sizes = [str(size) for size in args.sizes] if args.sizes else None

    if "refusal" in args.metrics:
        run_refusal_judge(data, args, sizes)
    if "correctness" in args.metrics:
        run_correctness(data, args, sizes)
    if "faithfulness" in args.metrics:
        run_faithfulness(data, args, sizes)

    save_json(args.output, data)
    print(f"Saved scored scaling results to {args.output}")


if __name__ == "__main__":
    main()
