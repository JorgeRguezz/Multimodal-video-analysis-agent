from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_build._llm import local_llm_config

from . import config
from .loop_profiles import get_loop_profile
from .models import (
    GapSnapshot,
    KnowledgeGap,
    SearchIntent,
    SearchPlanSnapshot,
    SearchPlanWarning,
)
from .prompts import SYSTEM_SEARCH_PLANNER, USER_SEARCH_PLAN_TEMPLATE


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: dict[str, Any], pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2 if pretty else None, ensure_ascii=False)


def load_gap_snapshot(path: str | Path) -> GapSnapshot:
    return GapSnapshot.model_validate(_read_json(Path(path)))


def select_target_gap(snapshot: GapSnapshot, gap_id: str | None = None) -> KnowledgeGap:
    gaps = snapshot.gaps
    if gap_id:
        for gap in gaps:
            if gap.gap_id == gap_id:
                return gap
        raise ValueError(f"Gap id not found: {gap_id}")

    open_gaps = [gap for gap in gaps if gap.status == "open"]
    if not open_gaps:
        raise ValueError("No open gaps available for search planning")
    return sorted(open_gaps, key=lambda gap: (-gap.priority, gap.topic_key, gap.facet))[0]


def _format_prompt(
    *,
    gap: KnowledgeGap,
    profile_id: str,
    top_n_intents: int,
) -> str:
    profile = get_loop_profile(profile_id)
    return USER_SEARCH_PLAN_TEMPLATE.format(
        profile_id=profile.id,
        profile_display_name=profile.display_name,
        extraction_game_id=profile.extraction_game_id,
        gap_id=gap.gap_id,
        gap_kind=gap.gap_kind,
        topic_key=gap.topic_key,
        facet=gap.facet,
        priority=gap.priority,
        coverage_score=gap.coverage_score,
        coverage_gap_score=gap.coverage_gap_score,
        weak_redundancy_score=gap.weak_redundancy_score,
        missing_facet_score=gap.missing_facet_score,
        freshness_score=gap.freshness_score,
        actionability_score=gap.actionability_score,
        evidence_summary=gap.evidence_summary,
        reasons=", ".join(gap.reasons) if gap.reasons else "none",
        top_n_intents=top_n_intents,
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("empty LLM output")

    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return data
        raise ValueError("top-level JSON is not an object")
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start < 0:
        raise ValueError("no JSON object start found")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(stripped)):
        char = stripped[idx]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : idx + 1]
                data = json.loads(candidate)
                if not isinstance(data, dict):
                    raise ValueError("extracted JSON is not an object")
                return data
    raise ValueError("no complete JSON object found")


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", str(query or "")).strip()


def _parse_search_intents(
    llm_text: str,
    *,
    top_n_intents: int,
) -> tuple[list[SearchIntent], list[SearchPlanWarning]]:
    warnings: list[SearchPlanWarning] = []
    try:
        payload = _extract_json_object(llm_text)
    except Exception as exc:  # noqa: BLE001
        return [], [
            SearchPlanWarning(
                code="llm_json_parse_failed",
                message=f"Could not parse search-plan JSON: {exc}",
            )
        ]

    raw_intents = payload.get("search_intents", [])
    if not isinstance(raw_intents, list):
        return [], [
            SearchPlanWarning(
                code="invalid_search_intents",
                message="LLM output field 'search_intents' is not a list",
            )
        ]

    intents: list[SearchIntent] = []
    seen_queries: set[str] = set()
    for idx, raw_intent in enumerate(raw_intents):
        if not isinstance(raw_intent, dict):
            warnings.append(
                SearchPlanWarning(
                    code="invalid_intent_entry",
                    message=f"Intent at index {idx} is not an object",
                )
            )
            continue
        raw_intent = dict(raw_intent)
        raw_intent["query"] = _normalize_query(raw_intent.get("query", ""))
        query_key = raw_intent["query"].lower()
        if not query_key:
            warnings.append(
                SearchPlanWarning(
                    code="empty_query_removed",
                    message=f"Intent at index {idx} had an empty query",
                )
            )
            continue
        if query_key in seen_queries:
            warnings.append(
                SearchPlanWarning(
                    code="duplicate_query_removed",
                    message=f"Duplicate query removed: {raw_intent['query']}",
                )
            )
            continue
        try:
            intent = SearchIntent.model_validate(raw_intent)
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                SearchPlanWarning(
                    code="intent_validation_failed",
                    message=f"Intent at index {idx} failed validation: {exc}",
                )
            )
            continue
        intents.append(intent)
        seen_queries.add(query_key)
        if len(intents) >= top_n_intents:
            break

    if not intents:
        warnings.append(
            SearchPlanWarning(
                code="no_valid_intents",
                message="No valid search intents were produced",
            )
        )
    return intents, warnings


async def build_search_plan(
    gap_snapshot: GapSnapshot,
    *,
    gap_id: str | None = None,
    profile_id: str | None = None,
    top_n_intents: int = 5,
    keep_raw_output: bool = False,
) -> SearchPlanSnapshot:
    resolved_profile_id = profile_id or gap_snapshot.loop_profile_id
    profile = get_loop_profile(resolved_profile_id)
    target_gap = select_target_gap(gap_snapshot, gap_id=gap_id)
    prompt = _format_prompt(
        gap=target_gap,
        profile_id=profile.id,
        top_n_intents=top_n_intents,
    )

    warnings: list[SearchPlanWarning] = []
    if gap_snapshot.loop_profile_id != profile.id:
        warnings.append(
            SearchPlanWarning(
                code="profile_mismatch",
                message=(
                    f"Gap snapshot profile {gap_snapshot.loop_profile_id!r} does not match "
                    f"search-plan profile {profile.id!r}"
                ),
            )
        )

    try:
        result = await local_llm_config.best_model_func(
            prompt,
            system_prompt=SYSTEM_SEARCH_PLANNER,
            max_tokens=1200,
            temperature=0.2,
            top_p=1.0,
            top_k=0,
            repeat_penalty=1.08,
            return_metadata=True,
        )
    except Exception as exc:  # noqa: BLE001
        return SearchPlanSnapshot(
            generated_at=datetime.now(timezone.utc),
            loop_profile_id=profile.id,
            target_gap=target_gap,
            search_intents=[],
            warnings=warnings
            + [
                SearchPlanWarning(
                    code="llm_call_failed",
                    message=f"Local LLM call failed: {exc}",
                )
            ],
            raw_llm_output=None,
        )

    if isinstance(result, dict):
        llm_answer = str(result.get("answer", ""))
        raw_output = str(result.get("raw_text", "") or llm_answer)
    else:
        llm_answer = str(result or "")
        raw_output = llm_answer

    intents, parse_warnings = _parse_search_intents(llm_answer or raw_output, top_n_intents=top_n_intents)
    warnings.extend(parse_warnings)

    return SearchPlanSnapshot(
        generated_at=datetime.now(timezone.utc),
        loop_profile_id=profile.id,
        target_gap=target_gap,
        search_intents=intents,
        warnings=warnings,
        raw_llm_output=raw_output if keep_raw_output else None,
    )


def persist_search_plan_snapshot(
    snapshot: SearchPlanSnapshot,
    state_root: str | Path | None = None,
    output_path: str | Path | None = None,
    pretty: bool = True,
) -> list[Path]:
    resolved_state_root = Path(state_root or config.DEFAULT_STATE_ROOT).resolve()
    search_dir = resolved_state_root / config.SEARCH_PLANS_DIR_NAME
    generated = snapshot.generated_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    paths = [
        search_dir / f"search_plan_{generated}.json",
        search_dir / "latest_search_plan.json",
    ]
    if output_path:
        paths.append(Path(output_path).resolve())

    payload = snapshot.model_dump(mode="json")
    for path in paths:
        _write_json(path, payload, pretty=pretty)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate LLM search intents for a selected knowledge gap.")
    parser.add_argument(
        "--gaps",
        default=str(config.DEFAULT_STATE_ROOT / config.GAPS_DIR_NAME / "latest_gaps.json"),
        help="Gap snapshot JSON path",
    )
    parser.add_argument("--gap-id", default=None, help="Specific gap id to plan searches for")
    parser.add_argument("--profile", default=None, help="Loop knowledge profile id")
    parser.add_argument("--top-n-intents", type=int, default=5)
    parser.add_argument("--state-root", default=str(config.DEFAULT_STATE_ROOT))
    parser.add_argument("--output", default=None, help="Optional extra output JSON path")
    parser.add_argument("--keep-raw-output", action="store_true")
    parser.add_argument("--compact", action="store_true", help="Write compact JSON instead of pretty JSON")
    return parser


async def _main_async(args: argparse.Namespace) -> int:
    gap_snapshot = load_gap_snapshot(args.gaps)
    snapshot = await build_search_plan(
        gap_snapshot,
        gap_id=args.gap_id,
        profile_id=args.profile,
        top_n_intents=max(1, args.top_n_intents),
        keep_raw_output=args.keep_raw_output,
    )
    paths = persist_search_plan_snapshot(
        snapshot,
        state_root=args.state_root,
        output_path=args.output,
        pretty=not args.compact,
    )

    print(f"Search-plan profile: {snapshot.loop_profile_id}")
    print(f"Target gap: {snapshot.target_gap.gap_id} {snapshot.target_gap.topic_key} {snapshot.target_gap.facet}")
    print(f"Search intents: {len(snapshot.search_intents)}")
    print(f"Warnings: {len(snapshot.warnings)}")
    for intent in snapshot.search_intents:
        print(f"- {intent.query} [{intent.source_type}]")
    for warning in snapshot.warnings:
        print(f"WARNING {warning.code}: {warning.message}")
    for path in paths:
        print(f"Wrote: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

