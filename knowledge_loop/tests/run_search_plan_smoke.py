from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from knowledge_loop.search_plan import build_search_plan, persist_search_plan_snapshot
from knowledge_loop.models import GapSnapshot, KnowledgeGap


def _fixture_gaps() -> GapSnapshot:
    target_gap = KnowledgeGap(
        gap_id="gap-search-plan-fixture",
        gap_kind="low_coverage",
        topic_key="RUNES::AATROX",
        facet="runes",
        priority=0.48,
        status="open",
        coverage_score=0.33,
        coverage_gap_score=0.67,
        missing_facet_score=0.0,
        freshness_days=None,
        freshness_score=None,
        weak_redundancy_score=0.67,
        actionability_score=0.90,
        video_count=1,
        chunk_count=2,
        source_redundancy=1,
        graph_entity_count=1,
        supporting_videos=["fixture_video"],
        source_inventory_topic_keys=["RUNES::AATROX"],
        evidence_summary=(
            "RUNES::AATROX / runes is supported by 1 video, 2 chunks, "
            "and 1 graph entity."
        ),
        reasons=["low_coverage", "weak_redundancy", "freshness_unknown"],
    )
    return GapSnapshot(
        generated_at=datetime.now(timezone.utc),
        loop_profile_id="league_of_legends",
        inventory_generated_at=datetime.now(timezone.utc),
        raw_bucket_count=1,
        actionable_bucket_count=1,
        filtered_bucket_count=0,
        gap_count=1,
        gaps=[target_gap],
    )


async def _run() -> int:
    snapshot = await build_search_plan(
        _fixture_gaps(),
        gap_id="gap-search-plan-fixture",
        profile_id="league_of_legends",
        top_n_intents=3,
        keep_raw_output=True,
    )

    assert snapshot.target_gap.gap_id == "gap-search-plan-fixture"
    assert snapshot.search_intents, [warning.model_dump() for warning in snapshot.warnings]
    assert all(intent.query.strip() for intent in snapshot.search_intents)

    with tempfile.TemporaryDirectory(prefix="knowledge-loop-search-plan-") as tmp:
        state_root = Path(tmp)
        paths = persist_search_plan_snapshot(snapshot, state_root=state_root)
        assert (state_root / "search_plans" / "latest_search_plan.json").exists()
        assert len(paths) == 2
        with (state_root / "search_plans" / "latest_search_plan.json").open("r", encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["loop_profile_id"] == "league_of_legends"
        assert payload["search_intents"]

    print("search-plan smoke test passed")
    for intent in snapshot.search_intents:
        print(f"- {intent.query} [{intent.source_type}]")
    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())

