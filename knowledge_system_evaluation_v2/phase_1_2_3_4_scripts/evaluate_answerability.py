from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_build._llm import (  # noqa: E402
    OSS_MODEL_ID,
    local_llm_config,
    shutdown_all_llm_resources,
)
from knowledge_inference.query_analyzer import analyze_query  # noqa: E402
from knowledge_inference.reranker import (  # noqa: E402
    apply_weighted_score,
    compute_component_scores,
    dedupe_hits,
)
from knowledge_inference.retrievers import retrieve_all  # noqa: E402
from knowledge_inference.service import InferenceService  # noqa: E402
from knowledge_inference.types import RetrievalHit  # noqa: E402

BASE_DIR = PROJECT_ROOT / "knowledge_system_evaluation_v2"
DEFAULT_INPUT = BASE_DIR / "evaluated_datasets" / "community_qa_dataset_final_v2.json"
DEFAULT_OUTPUT = BASE_DIR / "evaluated_datasets" / "community_qa_dataset_final_v3.json"
METHOD_VERSION = "claim_evidence_v3"
CODE_FENCE = chr(96) * 3
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_'-]+")
logger = logging.getLogger(__name__)

CLAIM_SYSTEM_PROMPT = """
You are an expert evaluation annotator for a video-derived League of Legends
knowledge base.

Reasoning: medium

Create a concise rubric of the essential factual requirements needed to answer the
question. Use the gold answer to disambiguate the intended scope, but do not require
its exact wording or incidental details. Accept semantically equivalent answers.
Do not use outside League of Legends knowledge and do not decide whether the
knowledge base supports the claims yet.

Return one JSON object in the final channel:
{
  "claims": [
    {"claim_id": "C1", "text": "...", "search_query": "..."}
  ],
  "requires_review": false,
  "review_reason": ""
}

Use between 1 and 10 essential claims. Set requires_review only for an unresolved
temporal, version, or interpretation ambiguity.

<|channel|>analysis<|message|>Identify only the essential answer requirements and
format them as JSON.<|end|>
<|start|>assistant<|channel|>final<|message|>{"claims":[{"claim_id":"C1",
"text":"The essential fact needed by the question.","search_query":"essential fact"}],
"requires_review":false,"review_reason":""}<|return|>
""".strip()

EVIDENCE_SYSTEM_PROMPT = """
You are an expert evidence annotator for a frozen, video-derived League of Legends
knowledge base.

Reasoning: medium

Judge only the supplied evidence. Do not use outside knowledge. For every claim,
choose exactly one status:
- supported: evidence explicitly states or unambiguously entails the claim;
- contradicted: evidence explicitly conflicts with the claim;
- insufficient: evidence is related, vague, incomplete, or does not establish it.

Evidence for a claim may be distributed across excerpts. Cite only supplied evidence
IDs. Do not treat retrieval relevance as factual support.

Return one JSON object in the final channel:
{
  "decisions": [
    {
      "claim_id": "C1",
      "status": "supported|contradicted|insufficient",
      "evidence_ids": ["E001"],
      "contributing_evidence_ids": [],
      "reason": "short evidence-grounded explanation"
    }
  ],
  "requires_review": false,
  "review_reason": ""
}

Return one decision for every claim. Set requires_review only for genuine temporal
or version conflict, irreducible ambiguity, or internally conflicting evidence.
For supported or contradicted, evidence_ids must contain at least one supplied ID.
For insufficient, use contributing_evidence_ids only for excerpts that establish a
material part of the claim and could combine with other excerpts; topical relevance
alone is not enough. Evidence may be combined across the supplied excerpts.

<|channel|>analysis<|message|>Compare every claim with only the supplied evidence
and return grounded JSON decisions.<|end|>
<|start|>assistant<|channel|>final<|message|>{"decisions":[{"claim_id":"C1",
"status":"supported","evidence_ids":["E001"],"contributing_evidence_ids":[],
"reason":"E001 states the claim."}],
"requires_review":false,"review_reason":""}<|return|>
""".strip()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def build_question(entry: dict[str, Any]) -> str:
    parts = [
        str(entry.get("question_title", "")).strip(),
        str(entry.get("question_body", "")).strip(),
    ]
    return "\n".join(part for part in parts if part)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip().replace("<|return|>", "")
    if cleaned.startswith(CODE_FENCE):
        cleaned = cleaned[len(CODE_FENCE) :].lstrip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
    if cleaned.endswith(CODE_FENCE):
        cleaned = cleaned[: -len(CODE_FENCE)].rstrip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Model response does not contain a JSON object")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Model response JSON must be an object")
    return payload


async def call_local_json(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Use GPT-OSS and the same chat formatting as segment_summarization_server."""
    last_error: Exception | None = None
    for attempt in range(2):
        prompt = user_prompt
        if attempt:
            prompt += (
                "\n\nYour previous response was not valid JSON. Return only the "
                "requested JSON object in the final channel."
            )
        result = await local_llm_config.best_model_func(
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=1.0,
            top_k=0,
            repeat_penalty=1.05,
            return_metadata=True,
        )
        try:
            return parse_json_object(str(result.get("answer", "")))
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            logger.warning("Malformed JSON response on attempt %s: %s", attempt + 1, exc)
    raise RuntimeError(f"GPT-OSS did not return valid JSON: {last_error}")


async def decompose_claims(
    question: str,
    gold_answer: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    prompt = f"""Question:
{question}

Gold answer used only to define the intended answer scope:
{gold_answer}

Produce the essential answer requirements now."""
    payload = await call_local_json(CLAIM_SYSTEM_PROMPT, prompt, 1800)

    claims: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in payload.get("claims", []):
        if isinstance(item, str):
            text, search_query = item.strip(), item.strip()
        elif isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            search_query = str(item.get("search_query", text)).strip() or text
        else:
            continue
        normalized = " ".join(text.lower().split())
        if not text or normalized in seen:
            continue
        seen.add(normalized)
        claims.append(
            {
                "claim_id": f"C{len(claims) + 1}",
                "text": text,
                "search_query": search_query,
            }
        )
        if len(claims) == 10:
            break

    if not claims:
        claims = [
            {
                "claim_id": "C1",
                "text": gold_answer.strip() or question,
                "search_query": question,
            }
        ]
        payload["requires_review"] = True
        payload["review_reason"] = "The judge returned no usable claim decomposition."

    metadata = {
        "requires_review": bool(payload.get("requires_review", False)),
        "review_reason": str(payload.get("review_reason", "")).strip(),
    }
    return claims, metadata


def build_bm25_index(
    stores: dict[str, Any],
) -> tuple[BM25Okapi, list[tuple[Any, str, str, list[str]]]]:
    corpus: list[list[str]] = []
    references: list[tuple[Any, str, str, list[str]]] = []
    for store in stores.values():
        for chunk_id, chunk in store.chunks_kv.items():
            text = str(chunk.get("content", "")).strip()
            if not text:
                continue
            segment_ids = chunk.get("video_segment_id", [])
            if isinstance(segment_ids, str):
                segment_ids = [segment_ids]
            corpus.append(tokenize(text))
            references.append(
                (store, str(chunk_id), text, [str(value) for value in segment_ids])
            )
    if not corpus:
        raise RuntimeError("The sanitized knowledge base contains no chunks")
    return BM25Okapi(corpus), references


def retrieve_bm25(
    query: str,
    bm25: BM25Okapi,
    references: list[tuple[Any, str, str, list[str]]],
    top_k: int,
) -> list[RetrievalHit]:
    scores = bm25.get_scores(tokenize(query))
    indices = sorted(
        range(len(scores)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )[:top_k]
    indices = [index for index in indices if float(scores[index]) > 0]
    maximum = max((float(scores[index]) for index in indices), default=0.0)
    hits: list[RetrievalHit] = []
    for index in indices:
        store, chunk_id, text, segment_ids = references[index]
        hits.append(
            RetrievalHit(
                chunk_id=chunk_id,
                video_name=store.video_name,
                source="bm25",
                chunk_text=text,
                segment_ids=segment_ids,
                score_semantic=float(scores[index]) / maximum if maximum else 0.0,
            )
        )
    return hits


async def collect_candidate_pool(
    question: str,
    claims: list[dict[str, str]],
    service: InferenceService,
    bm25: BM25Okapi,
    references: list[tuple[Any, str, str, list[str]]],
    args: argparse.Namespace,
) -> tuple[list[RetrievalHit], list[str]]:
    queries = [question] + [
        f"{question}\nEvidence requirement: {claim['search_query']}" for claim in claims
    ]
    queries = list(dict.fromkeys(queries))
    hits: list[RetrievalHit] = []
    graph = service.global_graph if service.global_graph is not None else nx.Graph()

    for query in queries:
        intent = analyze_query(query)
        hits.extend(
            await retrieve_all(
                query=query,
                intent=intent,
                stores=service.stores,
                global_graph=graph,
            )
        )
        hits.extend(retrieve_bm25(query, bm25, references, args.bm25_k))

    unique = dedupe_hits(hits)
    if not unique:
        return [], queries
    scored = compute_component_scores(unique, question, analyze_query(question))
    scored = apply_weighted_score(scored)
    scored.sort(key=lambda hit: hit.final_score, reverse=True)
    return scored[: args.candidate_pool_size], queries


def make_evidence(position: int, hit: RetrievalHit) -> dict[str, Any]:
    return {
        "evidence_id": f"E{position:03d}",
        "chunk_id": hit.chunk_id,
        "video_name": hit.video_name,
        "segment_ids": hit.segment_ids,
        "source": hit.source,
        "final_score": round(float(hit.final_score), 6),
        "text": hit.chunk_text,
    }


def render_evidence(records: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for record in records:
        text = str(record["text"])
        if len(text) > 5000:
            text = f"{text[:5000]}\n[excerpt truncated]"
        blocks.append(
            f"""[{record['evidence_id']}]
Video: {record['video_name']}
Chunk: {record['chunk_id']}
Retrieval source: {record['source']}
Content: {text}"""
        )
    return "\n\n--- NEXT EVIDENCE EXCERPT ---\n\n".join(blocks)


def validate_batch_payload(
    payload: dict[str, Any],
    claims: list[dict[str, str]],
    valid_evidence: set[str],
    stage: str,
    batch_number: int,
) -> tuple[list[dict[str, Any]], list[str], set[str]]:
    """Normalize one judge response and identify semantic schema violations."""
    expected_claims = [claim["claim_id"] for claim in claims]
    expected_set = set(expected_claims)
    violations: list[str] = []
    fatal_claims: set[str] = set()
    occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_decisions = payload.get("decisions", [])

    if not isinstance(raw_decisions, list):
        violations.append("decisions must be a list")
        fatal_claims.update(expected_set)
        raw_decisions = []

    for item in raw_decisions:
        if not isinstance(item, dict):
            violations.append("every decision must be an object")
            continue
        claim_id = str(item.get("claim_id", "")).strip()
        if claim_id not in expected_set:
            violations.append(f"unexpected claim_id {claim_id or '<empty>'}")
            continue
        occurrences[claim_id].append(item)

    normalized: list[dict[str, Any]] = []
    for claim_id in expected_claims:
        items = occurrences.get(claim_id, [])
        if len(items) != 1:
            problem = "missing" if not items else "duplicated"
            violations.append(f"{claim_id} is {problem}")
            fatal_claims.add(claim_id)
            normalized.append(
                {
                    "claim_id": claim_id,
                    "status": "insufficient",
                    "evidence_ids": [],
                    "contributing_evidence_ids": [],
                    "reason": f"The judge response was {problem} for this claim.",
                    "stage": stage,
                    "batch_number": batch_number,
                }
            )
            continue

        item = items[0]
        status = str(item.get("status", "insufficient")).lower().strip()
        if status not in {"supported", "contradicted", "insufficient"}:
            violations.append(f"{claim_id} has invalid status {status or '<empty>'}")
            fatal_claims.add(claim_id)
            status = "insufficient"

        def normalize_ids(field: str, fatal_on_invalid: bool) -> list[str]:
            values = item.get(field, [])
            if not isinstance(values, list):
                violations.append(f"{claim_id}.{field} must be a list")
                if fatal_on_invalid:
                    fatal_claims.add(claim_id)
                return []
            normalized_ids = list(dict.fromkeys(str(value) for value in values))
            invalid = [value for value in normalized_ids if value not in valid_evidence]
            if invalid:
                violations.append(
                    f"{claim_id}.{field} contains invalid IDs: {', '.join(invalid)}"
                )
                if fatal_on_invalid:
                    fatal_claims.add(claim_id)
            return [value for value in normalized_ids if value in valid_evidence]

        evidence_ids = normalize_ids("evidence_ids", fatal_on_invalid=True)
        contributing_ids = normalize_ids(
            "contributing_evidence_ids", fatal_on_invalid=False
        )
        if status in {"supported", "contradicted"} and not evidence_ids:
            violations.append(f"{claim_id} is {status} without a valid evidence ID")
            fatal_claims.add(claim_id)

        normalized.append(
            {
                "claim_id": claim_id,
                "status": status,
                "evidence_ids": evidence_ids,
                "contributing_evidence_ids": contributing_ids,
                "reason": str(item.get("reason", "")).strip(),
                "stage": stage,
                "batch_number": batch_number,
            }
        )
    return normalized, violations, fatal_claims


async def judge_batches(
    question: str,
    claims: list[dict[str, str]],
    evidence: list[dict[str, Any]],
    batch_size: int,
    stage: str,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    review_reasons: list[str] = []
    claim_text = "\n".join(f"- {c['claim_id']}: {c['text']}" for c in claims)
    diagnostics: dict[str, Any] = {
        "semantic_retry_count": 0,
        "semantic_failure_count": 0,
        "validation_events": [],
    }

    for offset in range(0, len(evidence), batch_size):
        batch = evidence[offset : offset + batch_size]
        valid_evidence = {item["evidence_id"] for item in batch}
        batch_number = offset // batch_size + 1
        consolidation_instruction = (
            "\nConsider the supplied excerpts jointly; complementary excerpts may "
            "establish the claim together."
            if stage == "consolidation"
            else ""
        )
        prompt = f"""Question:
{question}

Claims to judge:
{claim_text}

Evidence excerpts:
{render_evidence(batch)}

Judge every listed claim against this evidence batch.{consolidation_instruction}"""
        retry_violations: list[str] = []
        normalized: list[dict[str, Any]] = []
        fatal_claims: set[str] = set()
        payload: dict[str, Any] = {}
        for semantic_attempt in range(2):
            attempt_prompt = prompt
            if semantic_attempt:
                attempt_prompt += (
                    "\n\nYour previous JSON was semantically invalid:\n- "
                    + "\n- ".join(retry_violations)
                    + "\nReturn exactly one decision per claim. Supported and "
                    "contradicted decisions must cite at least one supplied evidence ID."
                )
            payload = await call_local_json(EVIDENCE_SYSTEM_PROMPT, attempt_prompt, 2400)
            normalized, violations, fatal_claims = validate_batch_payload(
                payload,
                claims,
                valid_evidence,
                stage,
                batch_number,
            )
            if not violations:
                if semantic_attempt:
                    diagnostics["validation_events"].append(
                        {
                            "stage": stage,
                            "batch_number": batch_number,
                            "recovered": True,
                            "errors": retry_violations,
                        }
                    )
                break
            if not semantic_attempt:
                retry_violations = violations
                diagnostics["semantic_retry_count"] += 1
                continue

            diagnostics["semantic_failure_count"] += 1
            diagnostics["validation_events"].append(
                {
                    "stage": stage,
                    "batch_number": batch_number,
                    "recovered": False,
                    "errors": violations,
                }
            )
            review_reasons.append(
                f"{stage} batch {batch_number} failed semantic validation after retry: "
                + "; ".join(violations)
            )
            for decision in normalized:
                if decision["claim_id"] in fatal_claims:
                    decision.update(
                        {
                            "status": "insufficient",
                            "evidence_ids": [],
                            "reason": (
                                "The judge decision remained invalid after one "
                                "semantic retry."
                            ),
                        }
                    )

        if payload.get("requires_review"):
            review_reasons.append(
                str(payload.get("review_reason", "")).strip()
                or "Evidence judge requested review."
            )
        decisions.extend(normalized)
    return decisions, review_reasons, diagnostics


def aggregate(
    claims: list[dict[str, str]],
    decisions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for decision in decisions:
        grouped[decision["claim_id"]].append(decision)

    results: list[dict[str, Any]] = []
    conflicts: list[str] = []
    for claim in claims:
        judged = grouped.get(claim["claim_id"], [])
        supported = [item for item in judged if item["status"] == "supported"]
        contradicted = [item for item in judged if item["status"] == "contradicted"]
        status = "supported" if supported else "contradicted" if contradicted else "insufficient"
        if supported and contradicted:
            conflicts.append(
                f"{claim['claim_id']} has both supporting and contradicting evidence."
            )
        results.append(
            {
                **claim,
                "status": status,
                "supporting_evidence_ids": sorted(
                    {value for item in supported for value in item["evidence_ids"]}
                ),
                "contradicting_evidence_ids": sorted(
                    {value for item in contradicted for value in item["evidence_ids"]}
                ),
                "contributing_evidence_ids": sorted(
                    {
                        value
                        for item in judged
                        for value in item.get("contributing_evidence_ids", [])
                    }
                ),
                "judgments": judged,
            }
        )
    return results, conflicts


def merge_judge_diagnostics(
    target: dict[str, Any],
    update: dict[str, Any],
) -> None:
    target["semantic_retry_count"] += int(update["semantic_retry_count"])
    target["semantic_failure_count"] += int(update["semantic_failure_count"])
    target["validation_events"].extend(update["validation_events"])


def classify_answerability(
    supported: int,
    total_claims: int,
    has_review_reasons: bool,
    contradicted: int,
) -> tuple[str, bool]:
    status = (
        "Full"
        if supported == total_claims
        else "Partial"
        if supported
        else "None"
    )
    requires_review = has_review_reasons or contradicted > 0
    return status, requires_review


def select_consolidation_evidence(
    claim_result: dict[str, Any],
    evidence: list[dict[str, Any]],
    top_k: int,
    char_budget: int,
) -> list[dict[str, Any]]:
    candidate_ids = set(claim_result["contributing_evidence_ids"])
    candidate_ids.update(claim_result["contradicting_evidence_ids"])
    selected: list[dict[str, Any]] = []
    rendered_size = 0
    separator_size = len("\n\n--- NEXT EVIDENCE EXCERPT ---\n\n")

    for record in evidence:
        if record["evidence_id"] not in candidate_ids:
            continue
        block_size = len(render_evidence([record]))
        projected = rendered_size + block_size
        if selected:
            projected += separator_size
        if projected > char_budget:
            continue
        selected.append(record)
        rendered_size = projected
        if len(selected) == top_k:
            break
    return selected


async def evaluate_entry(
    entry: dict[str, Any],
    source_index: int,
    service: InferenceService,
    bm25: BM25Okapi,
    references: list[tuple[Any, str, str, list[str]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    question = build_question(entry)
    claims, decomposition = await decompose_claims(
        question,
        str(entry.get("answer_gold", "")).strip(),
    )
    hits, queries = await collect_candidate_pool(
        question, claims, service, bm25, references, args
    )
    evidence = [make_evidence(index, hit) for index, hit in enumerate(hits, start=1)]
    initial = evidence[: args.initial_judge_k]

    decisions: list[dict[str, Any]] = []
    review_reasons: list[str] = []
    judge_diagnostics: dict[str, Any] = {
        "semantic_retry_count": 0,
        "semantic_failure_count": 0,
        "validation_events": [],
    }
    consolidation_runs: list[dict[str, Any]] = []
    if decomposition["requires_review"]:
        review_reasons.append(
            decomposition["review_reason"] or "Claim decomposition requested review."
        )
    if initial:
        batch_decisions, batch_review, batch_diagnostics = await judge_batches(
            question,
            claims,
            initial,
            args.evidence_batch_size,
            stage="initial",
        )
        decisions.extend(batch_decisions)
        review_reasons.extend(batch_review)
        merge_judge_diagnostics(judge_diagnostics, batch_diagnostics)

    initial_results, conflicts = aggregate(claims, decisions)
    review_reasons.extend(conflicts)
    unresolved = {
        result["claim_id"] for result in initial_results if result["status"] != "supported"
    }
    expanded: list[dict[str, Any]] = []
    if unresolved and len(evidence) > len(initial):
        expanded = evidence[len(initial) :]
        unresolved_claims = [claim for claim in claims if claim["claim_id"] in unresolved]
        batch_decisions, batch_review, batch_diagnostics = await judge_batches(
            question,
            unresolved_claims,
            expanded,
            args.evidence_batch_size,
            stage="expanded",
        )
        decisions.extend(batch_decisions)
        review_reasons.extend(batch_review)
        merge_judge_diagnostics(judge_diagnostics, batch_diagnostics)

    pre_consolidation_results, conflicts = aggregate(claims, decisions)
    review_reasons.extend(conflicts)
    claims_by_id = {claim["claim_id"]: claim for claim in claims}
    for claim_result in pre_consolidation_results:
        if claim_result["status"] == "supported":
            continue
        consolidation_evidence = select_consolidation_evidence(
            claim_result,
            evidence,
            args.consolidation_k,
            args.consolidation_char_budget,
        )
        if len(consolidation_evidence) < 2:
            continue
        claim_id = claim_result["claim_id"]
        (
            consolidation_decisions,
            consolidation_review,
            consolidation_diagnostics,
        ) = await judge_batches(
            question,
            [claims_by_id[claim_id]],
            consolidation_evidence,
            len(consolidation_evidence),
            stage="consolidation",
        )
        decisions.extend(consolidation_decisions)
        review_reasons.extend(consolidation_review)
        merge_judge_diagnostics(judge_diagnostics, consolidation_diagnostics)
        consolidation_runs.append(
            {
                "claim_id": claim_id,
                "evidence_ids": [
                    record["evidence_id"] for record in consolidation_evidence
                ],
                "rendered_character_count": len(
                    render_evidence(consolidation_evidence)
                ),
            }
        )

    claim_results, conflicts = aggregate(claims, decisions)
    review_reasons.extend(conflicts)
    if not hits:
        review_reasons.append("No candidates were retrieved from the sanitized knowledge base.")
    review_reasons = list(dict.fromkeys(reason for reason in review_reasons if reason))

    supported = sum(result["status"] == "supported" for result in claim_results)
    contradicted = sum(result["status"] == "contradicted" for result in claim_results)
    score = supported / len(claim_results)
    status, requires_review = classify_answerability(
        supported,
        len(claim_results),
        bool(review_reasons),
        contradicted,
    )

    output = dict(entry)
    output["answerability_status"] = status
    output["answerability_score"] = round(score, 4)
    if args.include_details:
        judged_count = len(initial) + len(expanded)
        output["answerability_details"] = {
            "method_version": METHOD_VERSION,
            "judge_model": OSS_MODEL_ID,
            "source_dataset_index": source_index,
            "configuration": {
                "input": str(args.input),
                "sample_size": args.sample_size,
                "candidate_pool_size": args.candidate_pool_size,
                "initial_judge_k": args.initial_judge_k,
                "bm25_k": args.bm25_k,
                "evidence_batch_size": args.evidence_batch_size,
                "consolidation_k": args.consolidation_k,
                "consolidation_char_budget": args.consolidation_char_budget,
                "seed": args.seed,
            },
            "claims": claim_results,
            "claim_decomposition": decomposition,
            "judge_validation": judge_diagnostics,
            "consolidation": {
                "claim_count": len(consolidation_runs),
                "runs": consolidation_runs,
            },
            "retrieval": {
                "knowledge_base_root": "knowledge_sanitization/cache",
                "search_queries": queries,
                "retrieval_methods": [
                    "bm25",
                    "dense_chunk",
                    "entity_graph",
                    "global_graph",
                    "visual_support",
                ],
                "candidate_pool_size": len(evidence),
                "initial_judge_count": len(initial),
                "expanded_judge_count": len(expanded),
                "judged_evidence_count": judged_count,
                "evidence": evidence[:judged_count],
            },
            "requires_review": requires_review,
            "review_reasons": review_reasons,
            "manual_review_priority": requires_review or status == "None",
        }
    return output


def select_entries(
    dataset: list[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> list[tuple[int, dict[str, Any]]]:
    if sample_size <= 0 or sample_size >= len(dataset):
        return list(enumerate(dataset))
    indices = sorted(random.Random(seed).sample(range(len(dataset)), sample_size))
    return [(index, dataset[index]) for index in indices]


def question_key(entry: dict[str, Any]) -> str:
    return str(entry.get("question_id") or build_question(entry))


async def process_dataset(args: argparse.Namespace) -> None:
    dataset = load_json(args.input)
    if not isinstance(dataset, list):
        raise ValueError(f"Expected a JSON list in {args.input}")
    selected = select_entries(dataset, args.sample_size, args.seed)

    if args.plan_only:
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Selected questions: {len(selected)} of {len(dataset)}")
        print(f"Include details: {args.include_details}")
        return
    if args.output.exists() and not args.resume:
        raise FileExistsError(
            f"Refusing to overwrite {args.output}. Use --resume or a new output path."
        )

    results: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.exists():
        existing = load_json(args.output)
        results = {question_key(item): item for item in existing}

    service = InferenceService()
    service.initialize()
    bm25, references = build_bm25_index(service.stores)

    try:
        for progress, (source_index, entry) in enumerate(selected, start=1):
            key = question_key(entry)
            if key in results and "answerability_status" in results[key]:
                logger.info("Skipping completed question %s (%s/%s)", key, progress, len(selected))
                continue
            logger.info(
                "Evaluating %s/%s, question %s: %s",
                progress,
                len(selected),
                key,
                entry.get("question_title", ""),
            )
            try:
                results[key] = await evaluate_entry(
                    entry, source_index, service, bm25, references, args
                )
            except Exception as exc:
                results[key] = {**entry, "answerability_error": str(exc)}
                ordered = [
                    results[question_key(item)]
                    for _, item in selected
                    if question_key(item) in results
                ]
                save_json(ordered, args.output)
                raise
            ordered = [
                results[question_key(item)]
                for _, item in selected
                if question_key(item) in results
            ]
            save_json(ordered, args.output)
    finally:
        shutdown_all_llm_resources()

    logger.info("Saved %s evaluated questions to %s", len(results), args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate corpus answerability using claim-level evidence judgments."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-pool-size", type=int, default=50)
    parser.add_argument("--initial-judge-k", type=int, default=20)
    parser.add_argument("--bm25-k", type=int, default=30)
    parser.add_argument("--evidence-batch-size", type=int, default=5)
    parser.add_argument("--consolidation-k", type=int, default=10)
    parser.add_argument("--consolidation-char-budget", type=int, default=30000)
    parser.add_argument("--include-details", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    if args.sample_size < 0:
        parser.error("--sample-size must be non-negative")
    if args.candidate_pool_size <= 0:
        parser.error("--candidate-pool-size must be positive")
    if not 0 < args.initial_judge_k <= args.candidate_pool_size:
        parser.error("--initial-judge-k must be within the candidate pool")
    if args.bm25_k <= 0 or args.evidence_batch_size <= 0:
        parser.error("--bm25-k and --evidence-batch-size must be positive")
    if args.consolidation_k <= 0:
        parser.error("--consolidation-k must be positive")
    if args.consolidation_char_budget <= 0:
        parser.error("--consolidation-char-budget must be positive")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(process_dataset(parse_args()))
