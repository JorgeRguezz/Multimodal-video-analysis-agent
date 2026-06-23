from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from knowledge_loop.gaps import detect_gaps, persist_gap_snapshot
from knowledge_loop.models import InventorySnapshot, InventorySourceRef, TopicFacetInventory, VideoInventory


def _row(
    *,
    topic_key: str,
    facet: str,
    video_count: int,
    chunk_count: int,
    coverage_score: float,
    weak_redundancy_score: float,
    graph_entity_count: int = 1,
) -> TopicFacetInventory:
    videos = [f"video_{idx}" for idx in range(video_count)]
    return TopicFacetInventory(
        topic_key=topic_key,
        facet=facet,
        video_count=video_count,
        chunk_count=chunk_count,
        graph_entity_count=graph_entity_count,
        source_redundancy=video_count,
        source_diversity_score=min(1.0, video_count / 3),
        weak_redundancy_score=weak_redundancy_score,
        coverage_score=coverage_score,
        freshness_days=None,
        supporting_videos=videos,
        evidence=[
            InventorySourceRef(
                source_kind="chunk",
                video_name=videos[0] if videos else "video_0",
                chunk_id=f"chunk-{topic_key.lower().replace('::', '-')}",
                segment_ids=[f"{videos[0]}_0"] if videos else [],
                text_preview=f"Fixture evidence for {topic_key}",
            )
        ],
    )


def _inventory() -> InventorySnapshot:
    return InventorySnapshot(
        generated_at=datetime.now(timezone.utc),
        loop_profile_id="league_of_legends",
        extraction_game_id="league_of_legends",
        source_root="/tmp/fixture-cache",
        videos={
            "video_0": VideoInventory(video_name="video_0", cache_dir="/tmp/video_0"),
            "video_1": VideoInventory(video_name="video_1", cache_dir="/tmp/video_1"),
            "video_2": VideoInventory(video_name="video_2", cache_dir="/tmp/video_2"),
        },
        topic_facets=[
            _row(
                topic_key="RUNES::AATROX",
                facet="runes",
                video_count=3,
                chunk_count=8,
                coverage_score=0.95,
                weak_redundancy_score=0.0,
            ),
            _row(
                topic_key="ITEMIZATION::AATROX",
                facet="itemization",
                video_count=1,
                chunk_count=2,
                coverage_score=0.35,
                weak_redundancy_score=0.6667,
            ),
            _row(
                topic_key="ABILITIES::E_ABILITY",
                facet="abilities",
                video_count=2,
                chunk_count=4,
                coverage_score=0.20,
                weak_redundancy_score=0.3333,
            ),
            _row(
                topic_key="MACRO::PYKE",
                facet="macro",
                video_count=1,
                chunk_count=1,
                coverage_score=0.15,
                weak_redundancy_score=0.6667,
            ),
            _row(
                topic_key="ABILITIES::AHRI",
                facet="abilities",
                video_count=2,
                chunk_count=4,
                coverage_score=0.55,
                weak_redundancy_score=0.3333,
            ),
        ],
    )


def main() -> int:
    inventory = _inventory()
    snapshot = detect_gaps(inventory, include_missing_facets=False)
    gap_keys = {(gap.topic_key, gap.facet, gap.gap_kind) for gap in snapshot.gaps}

    assert ("ITEMIZATION::AATROX", "itemization", "low_coverage") in gap_keys
    assert ("RUNES::AATROX", "runes", "low_coverage") not in gap_keys
    assert ("ABILITIES::E_ABILITY", "abilities", "low_coverage") not in gap_keys
    assert ("MACRO::PYKE", "macro", "low_coverage") not in gap_keys
    assert all(gap.freshness_score is None for gap in snapshot.gaps)

    missing_snapshot = detect_gaps(inventory, include_missing_facets=True)
    missing_keys = {(gap.topic_key, gap.facet, gap.gap_kind) for gap in missing_snapshot.gaps}
    assert ("RUNES::AHRI", "runes", "missing_facet") in missing_keys

    with tempfile.TemporaryDirectory(prefix="knowledge-loop-gaps-") as tmp:
        state_root = Path(tmp)
        paths = persist_gap_snapshot(missing_snapshot, state_root=state_root)
        assert (state_root / "gaps" / "latest_gaps.json").exists()
        assert len(paths) == 2
        with (state_root / "gaps" / "latest_gaps.json").open("r", encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["loop_profile_id"] == "league_of_legends"
        assert payload["gap_count"] == len(missing_snapshot.gaps)

    print("gaps fixture test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

