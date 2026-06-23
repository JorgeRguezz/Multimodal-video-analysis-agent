from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from . import config
from .loop_profiles import get_loop_profile
from .models import (
    InventorySnapshot,
    InventorySourceRef,
    InventoryWarning,
    LoopGameKnowledgeProfile,
    TopicFacetInventory,
    VideoInventory,
)


REQUIRED_INVENTORY_FILES = (
    "kv_store_text_chunks.json",
    "kv_store_video_segments.json",
    "kv_store_video_frames.json",
    "kv_store_video_path.json",
    "graph_chunk_entity_relation_clean.graphml",
)

GRAPH_FIELD_SEP = "<SEP>"
MAX_EVIDENCE_PER_BUCKET = 8


@dataclass
class _Bucket:
    topic_key: str
    facet: str
    videos: set[str] = field(default_factory=set)
    chunks: set[str] = field(default_factory=set)
    graph_nodes: set[str] = field(default_factory=set)
    evidence: list[InventorySourceRef] = field(default_factory=list)


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
            sort_keys=False,
            ensure_ascii=False,
        )


def _preview(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 14].rstrip() + " [truncated]"


def _slug(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip().upper()
    value = re.sub(r"[^A-Z0-9]+", "_", value).strip("_")
    return value or "UNKNOWN"


def _normalize_entity(value: Any, profile: LoopGameKnowledgeProfile) -> str | None:
    if value is None:
        return None
    raw = re.sub(r"\s+", " ", str(value)).strip().strip("\"'")
    if not raw:
        return None
    slug = _slug(raw)
    if slug in {_slug(item) for item in profile.entity_stopwords}:
        return None
    if len(slug) > 120:
        return None
    return slug


def _topic_key_for_entity(
    profile: LoopGameKnowledgeProfile,
    entity_slug: str | None,
    facet: str,
) -> str:
    if entity_slug:
        prefix = profile.topic_prefix_by_facet.get(facet)
        if not prefix:
            prefix = profile.topic_prefix_by_facet.get(profile.default_facet, "ENTITY")
        return f"{prefix}::{entity_slug}"
    return profile.generic_topic_key or f"GENERAL::{_slug(profile.id)}"


def _classify_facets(profile: LoopGameKnowledgeProfile, text: str) -> list[str]:
    lowered = f" {str(text or '').lower()} "
    scores: dict[str, int] = {}
    for facet, keywords in profile.facet_keywords.items():
        score = 0
        for keyword in keywords:
            needle = str(keyword).lower()
            if needle and needle in lowered:
                score += 1
        if score > 0:
            scores[facet] = score

    if not scores:
        return [profile.default_facet]

    return [
        facet
        for facet, _score in sorted(
            scores.items(),
            key=lambda item: (-item[1], profile.facets.index(item[0])),
        )
    ]


def _safe_segment_ids(chunk: dict[str, Any], video_name: str) -> list[str]:
    raw = chunk.get("video_segment_id", [])
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    segment_ids = []
    for value in raw:
        if not isinstance(value, str):
            continue
        if value.startswith(f"{video_name}_"):
            segment_ids.append(value)
    return sorted(set(segment_ids))


def _segment_index_from_ref(video_name: str, segment_ref: str) -> str | None:
    prefix = f"{video_name}_"
    if not segment_ref.startswith(prefix):
        return None
    return segment_ref[len(prefix) :]


def _segment_time_span(
    video_name: str,
    segment_ids: list[str],
    segments_map: dict[str, Any],
) -> str | None:
    starts: list[float] = []
    ends: list[float] = []
    for segment_id in segment_ids:
        idx = _segment_index_from_ref(video_name, segment_id)
        if idx is None:
            continue
        rec = segments_map.get(idx, {})
        raw_time = rec.get("time") if isinstance(rec, dict) else None
        if not isinstance(raw_time, str) or "-" not in raw_time:
            continue
        start_s, end_s = raw_time.split("-", 1)
        try:
            starts.append(float(start_s))
            ends.append(float(end_s))
        except ValueError:
            continue
    if not starts or not ends:
        return None
    return f"{min(starts):g}-{max(ends):g}"


def _split_source_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        candidates = value
    elif isinstance(value, str):
        candidates = value.split(GRAPH_FIELD_SEP)
    else:
        return []
    return sorted({str(item).strip() for item in candidates if str(item).strip()})


def _frame_entities_by_segment(
    video_name: str,
    frames_map: dict[str, Any],
    profile: LoopGameKnowledgeProfile,
) -> dict[str, set[str]]:
    by_segment: dict[str, set[str]] = defaultdict(set)
    use_lol_fields = profile.extraction_game_id == "league_of_legends"

    for frame_key, frame in frames_map.items():
        if not isinstance(frame, dict):
            continue
        seg_idx = str(frame.get("segment_idx", str(frame_key).split("_", 1)[0]))
        segment_ref = f"{video_name}_{seg_idx}"

        raw_entities: list[Any] = []
        entities = frame.get("entities", [])
        if isinstance(entities, list):
            raw_entities.extend(entities)

        if use_lol_fields:
            raw_entities.append(frame.get("main_champ"))
            partners = frame.get("partners", [])
            if isinstance(partners, list):
                raw_entities.extend(partners)

        for entity in raw_entities:
            normalized = _normalize_entity(entity, profile)
            if normalized:
                by_segment[segment_ref].add(normalized)

    return by_segment


def _declared_frame_games(frames_map: dict[str, Any]) -> set[str]:
    games: set[str] = set()
    for frame in frames_map.values():
        if not isinstance(frame, dict):
            continue
        game = frame.get("game")
        if isinstance(game, str) and game.strip():
            games.add(game.strip())
    return games


def _add_evidence(
    buckets: dict[tuple[str, str], _Bucket],
    profile: LoopGameKnowledgeProfile,
    video_name: str,
    facets: list[str],
    entities: set[str],
    source_ref: InventorySourceRef,
) -> None:
    if not facets:
        facets = [profile.default_facet]
    topic_entities = sorted(entities) if entities else [None]
    for facet in facets:
        for entity_slug in topic_entities:
            topic_key = _topic_key_for_entity(profile, entity_slug, facet)
            key = (topic_key, facet)
            bucket = buckets.setdefault(key, _Bucket(topic_key=topic_key, facet=facet))
            bucket.videos.add(video_name)
            if source_ref.chunk_id:
                bucket.chunks.add(source_ref.chunk_id)
            if source_ref.graph_node:
                bucket.graph_nodes.add(source_ref.graph_node)
            if len(bucket.evidence) < MAX_EVIDENCE_PER_BUCKET:
                bucket.evidence.append(source_ref)


def _load_graph(path: Path, warnings: list[InventoryWarning], video_name: str) -> nx.Graph:
    try:
        return nx.read_graphml(path)
    except Exception as exc:  # noqa: BLE001
        warnings.append(
            InventoryWarning(
                code="graph_load_failed",
                message=f"Could not load graph: {exc}",
                video_name=video_name,
                path=str(path),
            )
        )
        return nx.Graph()


def _video_name_from_cache_dir(video_dir: Path) -> str:
    name = video_dir.name
    if not name.startswith(config.SANITIZED_BUILD_PREFIX):
        return name
    return name[len(config.SANITIZED_BUILD_PREFIX) :]


def discover_sanitized_build_dirs(cache_root: Path) -> list[Path]:
    if not cache_root.exists():
        return []
    return sorted(path for path in cache_root.glob(config.SANITIZED_BUILD_GLOB) if path.is_dir())


def build_inventory_snapshot(
    profile_id: str | None = None,
    cache_root: str | Path | None = None,
) -> InventorySnapshot:
    profile = get_loop_profile(profile_id)
    resolved_cache_root = Path(cache_root or config.SANITIZED_CACHE_ROOT).resolve()
    warnings: list[InventoryWarning] = []
    videos: dict[str, VideoInventory] = {}
    buckets: dict[tuple[str, str], _Bucket] = {}

    build_dirs = discover_sanitized_build_dirs(resolved_cache_root)
    if not build_dirs:
        warnings.append(
            InventoryWarning(
                code="no_sanitized_build_caches",
                message=f"No sanitized build caches found under {resolved_cache_root}",
                path=str(resolved_cache_root),
            )
        )

    for video_dir in build_dirs:
        video_name = _video_name_from_cache_dir(video_dir)
        missing = [fname for fname in REQUIRED_INVENTORY_FILES if not (video_dir / fname).exists()]
        if missing:
            warnings.append(
                InventoryWarning(
                    code="missing_inventory_files",
                    message=f"Skipping cache with missing files: {missing}",
                    video_name=video_name,
                    path=str(video_dir),
                )
            )
            continue

        chunks = _read_json(video_dir / "kv_store_text_chunks.json")
        segments_root = _read_json(video_dir / "kv_store_video_segments.json")
        frames_root = _read_json(video_dir / "kv_store_video_frames.json")
        path_root = _read_json(video_dir / "kv_store_video_path.json")
        segments_map = segments_root.get(video_name, {}) if isinstance(segments_root, dict) else {}
        frames_map = frames_root.get(video_name, {}) if isinstance(frames_root, dict) else {}
        if not isinstance(segments_map, dict):
            segments_map = {}
        if not isinstance(frames_map, dict):
            frames_map = {}

        declared_games = _declared_frame_games(frames_map)
        if declared_games and profile.extraction_game_id not in declared_games:
            warnings.append(
                InventoryWarning(
                    code="profile_game_mismatch",
                    message=(
                        f"Skipping cache for declared games {sorted(declared_games)} "
                        f"while inventory profile expects {profile.extraction_game_id!r}"
                    ),
                    video_name=video_name,
                    path=str(video_dir),
                )
            )
            continue
        if not declared_games and profile.extraction_game_id != "league_of_legends":
            warnings.append(
                InventoryWarning(
                    code="unknown_video_game_for_non_default_profile",
                    message=(
                        "Skipping cache without explicit frame game metadata for "
                        f"non-default extraction game {profile.extraction_game_id!r}"
                    ),
                    video_name=video_name,
                    path=str(video_dir),
                )
            )
            continue
        if not declared_games:
            warnings.append(
                InventoryWarning(
                    code="unknown_video_game_assumed_legacy_lol",
                    message="Cache has no frame game metadata; treating it as legacy League of Legends data",
                    video_name=video_name,
                    path=str(video_dir),
                )
            )

        graph = _load_graph(video_dir / "graph_chunk_entity_relation_clean.graphml", warnings, video_name)
        videos[video_name] = VideoInventory(
            video_name=video_name,
            cache_dir=str(video_dir),
            source_path=path_root.get(video_name) if isinstance(path_root.get(video_name), str) else None,
            chunk_count=len(chunks),
            segment_count=len(segments_map),
            frame_count=len(frames_map),
            graph_node_count=graph.number_of_nodes(),
            graph_edge_count=graph.number_of_edges(),
            freshness_days=None,
        )

        entities_by_segment = _frame_entities_by_segment(video_name, frames_map, profile)
        chunk_segment_ids: dict[str, list[str]] = {}

        for chunk_id, chunk in chunks.items():
            if not isinstance(chunk, dict):
                continue
            chunk_text = str(chunk.get("content", ""))
            segment_ids = _safe_segment_ids(chunk, video_name)
            chunk_segment_ids[chunk_id] = segment_ids

            entities: set[str] = set()
            for segment_id in segment_ids:
                entities.update(entities_by_segment.get(segment_id, set()))

            facets = _classify_facets(profile, chunk_text)
            _add_evidence(
                buckets=buckets,
                profile=profile,
                video_name=video_name,
                facets=facets,
                entities=entities,
                source_ref=InventorySourceRef(
                    source_kind="chunk",
                    video_name=video_name,
                    chunk_id=chunk_id,
                    segment_ids=segment_ids,
                    time_span=_segment_time_span(video_name, segment_ids, segments_map),
                    text_preview=_preview(chunk_text),
                ),
            )

        for node_id, attrs in graph.nodes(data=True):
            if not isinstance(attrs, dict):
                attrs = {}
            node_entity = _normalize_entity(node_id, profile)
            if not node_entity:
                continue
            description = str(attrs.get("description", ""))
            graph_text = f"{node_id}\n{description}"
            facets = _classify_facets(profile, graph_text)
            source_ids = [sid for sid in _split_source_ids(attrs.get("source_id")) if sid in chunks]
            segment_ids = sorted(
                {
                    segment_id
                    for source_id in source_ids
                    for segment_id in chunk_segment_ids.get(source_id, [])
                }
            )
            _add_evidence(
                buckets=buckets,
                profile=profile,
                video_name=video_name,
                facets=facets,
                entities={node_entity},
                source_ref=InventorySourceRef(
                    source_kind="graph",
                    video_name=video_name,
                    chunk_id=source_ids[0] if source_ids else None,
                    segment_ids=segment_ids,
                    graph_node=str(node_id),
                    time_span=_segment_time_span(video_name, segment_ids, segments_map),
                    text_preview=_preview(graph_text),
                ),
            )

    topic_facets = [
        _bucket_to_inventory(bucket, profile)
        for bucket in sorted(buckets.values(), key=lambda item: (item.topic_key, item.facet))
    ]

    global_graph_path = resolved_cache_root / "sanitized_global" / "graph_AetherNexus.graphml"
    global_nodes: int | None = None
    global_edges: int | None = None
    if global_graph_path.exists():
        try:
            global_graph = nx.read_graphml(global_graph_path)
            global_nodes = global_graph.number_of_nodes()
            global_edges = global_graph.number_of_edges()
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                InventoryWarning(
                    code="global_graph_load_failed",
                    message=f"Could not load sanitized global graph: {exc}",
                    path=str(global_graph_path),
                )
            )

    return InventorySnapshot(
        generated_at=datetime.now(timezone.utc),
        loop_profile_id=profile.id,
        extraction_game_id=profile.extraction_game_id,
        source_root=str(resolved_cache_root),
        videos=dict(sorted(videos.items())),
        topic_facets=topic_facets,
        warnings=warnings,
        global_graph_node_count=global_nodes,
        global_graph_edge_count=global_edges,
    )


def _bucket_to_inventory(
    bucket: _Bucket,
    profile: LoopGameKnowledgeProfile,
) -> TopicFacetInventory:
    video_count = len(bucket.videos)
    chunk_count = len(bucket.chunks)
    source_diversity = min(1.0, video_count / max(1, profile.target_sources_per_facet))
    chunk_score = min(1.0, chunk_count / max(1, profile.target_chunks_per_facet))
    coverage = 0.70 * chunk_score + 0.30 * source_diversity
    return TopicFacetInventory(
        topic_key=bucket.topic_key,
        facet=bucket.facet,
        video_count=video_count,
        chunk_count=chunk_count,
        graph_entity_count=len(bucket.graph_nodes),
        source_redundancy=video_count,
        source_diversity_score=round(source_diversity, 4),
        weak_redundancy_score=round(1.0 - source_diversity, 4),
        coverage_score=round(coverage, 4),
        freshness_days=None,
        supporting_videos=sorted(bucket.videos),
        evidence=bucket.evidence,
    )


def persist_inventory_snapshot(
    snapshot: InventorySnapshot,
    state_root: str | Path | None = None,
    output_path: str | Path | None = None,
    pretty: bool = True,
) -> list[Path]:
    resolved_state_root = Path(state_root or config.DEFAULT_STATE_ROOT).resolve()
    inventory_dir = resolved_state_root / config.INVENTORY_DIR_NAME
    generated = snapshot.generated_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    paths = [
        inventory_dir / f"inventory_{generated}.json",
        inventory_dir / "latest_inventory.json",
    ]
    if output_path:
        paths.append(Path(output_path).resolve())

    payload = snapshot.model_dump(mode="json")
    for path in paths:
        _write_json(path, payload, pretty=pretty)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a knowledge-loop KB inventory snapshot.")
    parser.add_argument("--profile", default=None, help="Loop knowledge profile id")
    parser.add_argument("--cache-root", default=str(config.SANITIZED_CACHE_ROOT))
    parser.add_argument("--state-root", default=str(config.DEFAULT_STATE_ROOT))
    parser.add_argument("--output", default=None, help="Optional extra output JSON path")
    parser.add_argument("--compact", action="store_true", help="Write compact JSON instead of pretty JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    snapshot = build_inventory_snapshot(profile_id=args.profile, cache_root=args.cache_root)
    paths = persist_inventory_snapshot(
        snapshot,
        state_root=args.state_root,
        output_path=args.output,
        pretty=not args.compact,
    )

    print(f"Inventory profile: {snapshot.loop_profile_id}")
    print(f"Videos inventoried: {len(snapshot.videos)}")
    print(f"Topic/facet buckets: {len(snapshot.topic_facets)}")
    print(f"Warnings: {len(snapshot.warnings)}")
    for path in paths:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
