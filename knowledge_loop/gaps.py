from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config
from .loop_profiles import get_loop_profile
from .models import (
    GapSnapshot,
    GapWarning,
    InventorySnapshot,
    KnowledgeGap,
    LoopGameKnowledgeProfile,
    TopicFacetInventory,
)


@dataclass(frozen=True)
class TopicParts:
    prefix: str
    entity_key: str
    entity_tokens: tuple[str, ...]


@dataclass(frozen=True)
class Actionability:
    score: float
    is_actionable: bool
    reasons: tuple[str, ...]
    filtered_reason: str | None = None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: dict[str, Any], pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2 if pretty else None,
            ensure_ascii=False,
        )


def load_inventory_snapshot(path: str | Path) -> InventorySnapshot:
    return InventorySnapshot.model_validate(_read_json(Path(path)))


def _slug(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip().upper()
    value = re.sub(r"[^A-Z0-9]+", "_", value).strip("_")
    return value or "UNKNOWN"


def _parse_topic_key(topic_key: str) -> TopicParts | None:
    parts = [part for part in str(topic_key or "").split("::") if part]
    if len(parts) < 2:
        return None
    prefix = _slug(parts[0])
    entity_tokens = tuple(_slug(part) for part in parts[1:] if _slug(part))
    if not entity_tokens:
        return None
    return TopicParts(
        prefix=prefix,
        entity_key="::".join(entity_tokens),
        entity_tokens=entity_tokens,
    )


def _is_generic_topic(topic_key: str, profile: LoopGameKnowledgeProfile) -> bool:
    if topic_key.startswith("GENERAL::"):
        return True
    return bool(profile.generic_topic_key and topic_key == profile.generic_topic_key)


def _has_noisy_entity(parts: TopicParts, profile: LoopGameKnowledgeProfile) -> bool:
    noisy = {_slug(token) for token in profile.noisy_topic_tokens}
    return any(token in noisy for token in parts.entity_tokens)


def _camel_slug(value: str) -> str:
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    return _slug(spaced)


def _profile_missing_facet_entity_allowlist(profile: LoopGameKnowledgeProfile) -> set[str] | None:
    """Return entity slugs eligible for synthetic missing-facet gaps.

    For LoL, missing-facet gaps should apply to champions, not every graph entity
    such as BARON_PIT or BOTTOM_LANE_TURRET. Other profiles can add their own
    allowlist later; without one, missing-facet generation is skipped.
    """

    if profile.id != "league_of_legends":
        return None

    allowed: set[str] = set()
    champions_dir = config.PROJECT_ROOT / "knowledge_extraction" / "image_matching" / "assets" / "champions"
    if champions_dir.is_dir():
        for child in champions_dir.iterdir():
            if not child.is_dir():
                continue
            allowed.add(_slug(child.name))
            allowed.add(_camel_slug(child.name))

    alias_path = config.PROJECT_ROOT / "knowledge_sanitization" / "spec" / "alias_champions.json"
    if alias_path.exists():
        try:
            alias_data = _read_json(alias_path)
        except Exception:  # noqa: BLE001
            alias_data = {}
        for canonical, aliases in alias_data.items():
            allowed.add(_slug(canonical))
            allowed.add(_camel_slug(str(canonical)))
            if isinstance(aliases, list):
                for alias in aliases:
                    allowed.add(_slug(str(alias)))
                    allowed.add(_camel_slug(str(alias)))

    return allowed or None


def _freshness_score(
    freshness_days: int | None,
    profile: LoopGameKnowledgeProfile,
) -> float | None:
    if freshness_days is None:
        return None
    if freshness_days < 0:
        return None
    return min(1.0, freshness_days / max(1, profile.stale_after_days))


def _priority(
    *,
    coverage_gap_score: float,
    weak_redundancy_score: float,
    missing_facet_score: float,
    freshness_score: float | None,
    actionability_score: float,
) -> float:
    if freshness_score is None:
        base = (
            0.50 * coverage_gap_score
            + 0.30 * weak_redundancy_score
            + 0.20 * missing_facet_score
        )
    else:
        base = (
            0.40 * coverage_gap_score
            + 0.25 * freshness_score
            + 0.20 * weak_redundancy_score
            + 0.15 * missing_facet_score
        )
    return round(max(0.0, min(1.0, base * actionability_score)), 4)


def _gap_id(
    profile: LoopGameKnowledgeProfile,
    gap_kind: str,
    topic_key: str,
    facet: str,
) -> str:
    raw = f"{profile.id}|{gap_kind}|{topic_key}|{facet}".encode("utf-8")
    return "gap-" + hashlib.md5(raw).hexdigest()[:16]


def _evidence_summary(row: TopicFacetInventory) -> str:
    video_word = "video" if row.video_count == 1 else "videos"
    chunk_word = "chunk" if row.chunk_count == 1 else "chunks"
    return (
        f"{row.topic_key} / {row.facet} is supported by "
        f"{row.video_count} {video_word}, {row.chunk_count} {chunk_word}, "
        f"and {row.graph_entity_count} graph entities."
    )


def _assess_actionability(
    row: TopicFacetInventory,
    profile: LoopGameKnowledgeProfile,
    *,
    min_actionability: float | None = None,
) -> Actionability:
    threshold = min_actionability if min_actionability is not None else profile.min_actionability_score
    parts = _parse_topic_key(row.topic_key)

    if row.facet not in profile.facets:
        return Actionability(0.0, False, (), "unknown_facet")
    if _is_generic_topic(row.topic_key, profile):
        return Actionability(0.0, False, (), "generic_topic")
    if parts is None:
        return Actionability(0.0, False, (), "malformed_topic_key")
    if _has_noisy_entity(parts, profile):
        return Actionability(0.0, False, (), "noisy_topic_token")
    if not row.evidence:
        return Actionability(0.0, False, (), "no_evidence")
    if row.coverage_score >= profile.covered_threshold:
        return Actionability(0.0, False, (), "already_covered")
    if row.chunk_count < profile.min_actionable_chunks and row.video_count < profile.min_actionable_videos:
        return Actionability(0.0, False, (), "singleton_or_too_sparse")

    evidence_volume = min(1.0, row.chunk_count / max(1, profile.min_actionable_chunks))
    source_signal = min(1.0, row.video_count / max(1, profile.min_actionable_videos))
    graph_signal = 1.0 if row.graph_entity_count > 0 else 0.5
    score = 0.40 * evidence_volume + 0.30 * source_signal + 0.20 * graph_signal + 0.10
    score = round(max(0.0, min(1.0, score)), 4)

    reasons = ["actionable_topic"]
    if row.coverage_score < profile.covered_threshold:
        reasons.append("below_covered_threshold")
    if row.video_count < profile.target_sources_per_facet:
        reasons.append("low_source_diversity")
    if row.chunk_count < profile.target_chunks_per_facet:
        reasons.append("low_chunk_coverage")
    if row.graph_entity_count <= 0:
        reasons.append("no_graph_entity_support")

    if score < threshold:
        return Actionability(score, False, tuple(reasons), "low_actionability_score")
    return Actionability(score, True, tuple(reasons), None)


def _gap_from_inventory_row(
    row: TopicFacetInventory,
    profile: LoopGameKnowledgeProfile,
    actionability: Actionability,
) -> KnowledgeGap:
    coverage_gap_score = round(max(0.0, 1.0 - row.coverage_score), 4)
    fresh_score = _freshness_score(row.freshness_days, profile)
    reasons = list(actionability.reasons)
    if coverage_gap_score > 0.0:
        reasons.append("low_coverage")
    if row.weak_redundancy_score > 0.0:
        reasons.append("weak_redundancy")
    if fresh_score is None:
        reasons.append("freshness_unknown")
    elif fresh_score > 0.0:
        reasons.append("stale_coverage")

    priority = _priority(
        coverage_gap_score=coverage_gap_score,
        weak_redundancy_score=row.weak_redundancy_score,
        missing_facet_score=0.0,
        freshness_score=fresh_score,
        actionability_score=actionability.score,
    )

    return KnowledgeGap(
        gap_id=_gap_id(profile, "low_coverage", row.topic_key, row.facet),
        gap_kind="low_coverage",
        topic_key=row.topic_key,
        facet=row.facet,
        priority=priority,
        coverage_score=row.coverage_score,
        coverage_gap_score=coverage_gap_score,
        missing_facet_score=0.0,
        freshness_days=row.freshness_days,
        freshness_score=fresh_score,
        weak_redundancy_score=row.weak_redundancy_score,
        actionability_score=actionability.score,
        video_count=row.video_count,
        chunk_count=row.chunk_count,
        source_redundancy=row.source_redundancy,
        graph_entity_count=row.graph_entity_count,
        supporting_videos=row.supporting_videos,
        source_inventory_topic_keys=[row.topic_key],
        evidence_summary=_evidence_summary(row),
        reasons=sorted(set(reasons)),
    )


def _entity_rows(
    inventory: InventorySnapshot,
    profile: LoopGameKnowledgeProfile,
) -> dict[str, list[TopicFacetInventory]]:
    rows_by_entity: dict[str, list[TopicFacetInventory]] = {}
    missing_facet_allowlist = _profile_missing_facet_entity_allowlist(profile)
    for row in inventory.topic_facets:
        if _is_generic_topic(row.topic_key, profile):
            continue
        parts = _parse_topic_key(row.topic_key)
        if parts is None or _has_noisy_entity(parts, profile):
            continue
        if missing_facet_allowlist is None:
            continue
        if not all(token in missing_facet_allowlist for token in parts.entity_tokens):
            continue
        rows_by_entity.setdefault(parts.entity_key, []).append(row)
    return rows_by_entity


def _missing_facet_actionability(
    *,
    chunk_count: int,
    video_count: int,
    graph_entity_count: int,
    profile: LoopGameKnowledgeProfile,
    min_actionability: float | None,
) -> Actionability:
    threshold = min_actionability if min_actionability is not None else profile.min_actionability_score
    if chunk_count < profile.missing_facet_min_chunks and video_count < profile.missing_facet_min_videos:
        return Actionability(0.0, False, (), "entity_too_sparse_for_missing_facet")

    evidence_volume = min(1.0, chunk_count / max(1, profile.missing_facet_min_chunks))
    source_signal = min(1.0, video_count / max(1, profile.missing_facet_min_videos))
    graph_signal = 1.0 if graph_entity_count > 0 else 0.5
    score = 0.35 * evidence_volume + 0.35 * source_signal + 0.20 * graph_signal + 0.10
    score = round(max(0.0, min(1.0, score)), 4)
    reasons = ("known_entity_missing_core_facet", "synthetic_missing_facet")
    if score < threshold:
        return Actionability(score, False, reasons, "low_actionability_score")
    return Actionability(score, True, reasons, None)


def _generate_missing_facet_gaps(
    inventory: InventorySnapshot,
    profile: LoopGameKnowledgeProfile,
    *,
    min_actionability: float | None,
) -> list[KnowledgeGap]:
    gaps: list[KnowledgeGap] = []
    prefix_by_facet = profile.topic_prefix_by_facet

    for entity_key, rows in _entity_rows(inventory, profile).items():
        existing_facets = {row.facet for row in rows}
        videos = sorted({video for row in rows for video in row.supporting_videos})
        chunks = {evidence.chunk_id for row in rows for evidence in row.evidence if evidence.chunk_id}
        graph_nodes = {evidence.graph_node for row in rows for evidence in row.evidence if evidence.graph_node}
        source_topic_keys = sorted({row.topic_key for row in rows})

        actionability = _missing_facet_actionability(
            chunk_count=len(chunks),
            video_count=len(videos),
            graph_entity_count=len(graph_nodes),
            profile=profile,
            min_actionability=min_actionability,
        )
        if not actionability.is_actionable:
            continue

        for facet in profile.core_gap_facets:
            if facet in existing_facets:
                continue
            prefix = prefix_by_facet.get(facet)
            if not prefix:
                continue
            topic_key = f"{prefix}::{entity_key}"
            fresh_score = None
            coverage_gap_score = 1.0
            weak_redundancy = 1.0
            priority = _priority(
                coverage_gap_score=coverage_gap_score,
                weak_redundancy_score=weak_redundancy,
                missing_facet_score=1.0,
                freshness_score=fresh_score,
                actionability_score=actionability.score,
            )
            gaps.append(
                KnowledgeGap(
                    gap_id=_gap_id(profile, "missing_facet", topic_key, facet),
                    gap_kind="missing_facet",
                    topic_key=topic_key,
                    facet=facet,
                    priority=priority,
                    coverage_score=0.0,
                    coverage_gap_score=coverage_gap_score,
                    missing_facet_score=1.0,
                    freshness_days=None,
                    freshness_score=None,
                    weak_redundancy_score=weak_redundancy,
                    actionability_score=actionability.score,
                    video_count=0,
                    chunk_count=0,
                    source_redundancy=0,
                    graph_entity_count=0,
                    supporting_videos=videos,
                    source_inventory_topic_keys=source_topic_keys,
                    evidence_summary=(
                        f"{entity_key} appears in {len(videos)} videos and "
                        f"{len(chunks)} chunks, but the core facet {facet!r} is absent."
                    ),
                    reasons=sorted(set(actionability.reasons + ("missing_core_facet", "freshness_unknown"))),
                )
            )

    return gaps


def detect_gaps(
    inventory: InventorySnapshot,
    *,
    profile_id: str | None = None,
    include_missing_facets: bool = False,
    min_actionability: float | None = None,
    inventory_source_path: str | None = None,
) -> GapSnapshot:
    profile = get_loop_profile(profile_id or inventory.loop_profile_id)
    warnings: list[GapWarning] = []
    gaps: list[KnowledgeGap] = []
    actionable_count = 0
    filtered_count = 0

    if inventory.loop_profile_id != profile.id:
        warnings.append(
            GapWarning(
                code="profile_mismatch",
                message=(
                    f"Inventory profile {inventory.loop_profile_id!r} does not match "
                    f"gap profile {profile.id!r}"
                ),
            )
        )

    for row in inventory.topic_facets:
        actionability = _assess_actionability(row, profile, min_actionability=min_actionability)
        if not actionability.is_actionable:
            filtered_count += 1
            continue
        actionable_count += 1
        gaps.append(_gap_from_inventory_row(row, profile, actionability))

    if include_missing_facets:
        missing_gaps = _generate_missing_facet_gaps(
            inventory,
            profile,
            min_actionability=min_actionability,
        )
        gaps.extend(missing_gaps)
        actionable_count += len(missing_gaps)

    gaps = sorted(
        gaps,
        key=lambda gap: (
            -gap.priority,
            gap.gap_kind,
            gap.topic_key,
            gap.facet,
        ),
    )

    return GapSnapshot(
        generated_at=datetime.now(timezone.utc),
        loop_profile_id=profile.id,
        inventory_generated_at=inventory.generated_at,
        inventory_source_path=inventory_source_path,
        raw_bucket_count=len(inventory.topic_facets),
        actionable_bucket_count=actionable_count,
        filtered_bucket_count=filtered_count,
        gap_count=len(gaps),
        include_missing_facets=include_missing_facets,
        gaps=gaps,
        warnings=warnings,
    )


def persist_gap_snapshot(
    snapshot: GapSnapshot,
    state_root: str | Path | None = None,
    output_path: str | Path | None = None,
    pretty: bool = True,
) -> list[Path]:
    resolved_state_root = Path(state_root or config.DEFAULT_STATE_ROOT).resolve()
    gaps_dir = resolved_state_root / config.GAPS_DIR_NAME
    generated = snapshot.generated_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    paths = [
        gaps_dir / f"gaps_{generated}.json",
        gaps_dir / "latest_gaps.json",
    ]
    if output_path:
        paths.append(Path(output_path).resolve())

    payload = snapshot.model_dump(mode="json")
    for path in paths:
        _write_json(path, payload, pretty=pretty)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect ranked knowledge gaps from an inventory snapshot.")
    parser.add_argument(
        "--inventory",
        default=str(config.DEFAULT_STATE_ROOT / config.INVENTORY_DIR_NAME / "latest_inventory.json"),
        help="Inventory snapshot JSON path",
    )
    parser.add_argument("--profile", default=None, help="Loop knowledge profile id")
    parser.add_argument("--state-root", default=str(config.DEFAULT_STATE_ROOT))
    parser.add_argument("--output", default=None, help="Optional extra output JSON path")
    parser.add_argument("--top-n", type=int, default=None, help="Persist only the top N gaps")
    parser.add_argument("--include-missing-facets", action="store_true")
    parser.add_argument("--min-actionability", type=float, default=None)
    parser.add_argument("--compact", action="store_true", help="Write compact JSON instead of pretty JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    inventory_path = Path(args.inventory)
    inventory = load_inventory_snapshot(inventory_path)
    snapshot = detect_gaps(
        inventory,
        profile_id=args.profile,
        include_missing_facets=args.include_missing_facets,
        min_actionability=args.min_actionability,
        inventory_source_path=str(inventory_path.resolve()),
    )
    if args.top_n is not None:
        snapshot.gaps = snapshot.gaps[: max(0, args.top_n)]
        snapshot.gap_count = len(snapshot.gaps)

    paths = persist_gap_snapshot(
        snapshot,
        state_root=args.state_root,
        output_path=args.output,
        pretty=not args.compact,
    )

    print(f"Gap profile: {snapshot.loop_profile_id}")
    print(f"Raw inventory buckets: {snapshot.raw_bucket_count}")
    print(f"Actionable buckets: {snapshot.actionable_bucket_count}")
    print(f"Filtered buckets: {snapshot.filtered_bucket_count}")
    print(f"Gaps: {snapshot.gap_count}")
    print(f"Warnings: {len(snapshot.warnings)}")
    for gap in snapshot.gaps[:10]:
        print(f"{gap.priority:.4f} {gap.gap_kind} {gap.topic_key} {gap.facet}")
    for path in paths:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
