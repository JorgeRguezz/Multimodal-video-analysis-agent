from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from knowledge_loop.models import KnowledgeGap, SearchIntent, SearchPlanSnapshot
from knowledge_loop.youtube_search import persist_youtube_search_snapshot, run_youtube_search


def _search_plan() -> SearchPlanSnapshot:
    return SearchPlanSnapshot(
        generated_at=datetime.now(timezone.utc),
        loop_profile_id="league_of_legends",
        target_gap=KnowledgeGap(
            gap_id="gap-youtube-smoke",
            gap_kind="low_coverage",
            topic_key="RUNES::AATROX",
            facet="runes",
            priority=0.48,
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
            evidence_summary="RUNES::AATROX / runes is weakly covered.",
            reasons=["low_coverage", "weak_redundancy", "freshness_unknown"],
        ),
        search_intents=[
            SearchIntent(
                query="Aatrox rune guide current season",
                purpose="Find Aatrox rune setup explanations.",
                source_type="educational_guide",
            )
        ],
    )


async def _run() -> int:
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        print("Skipping real YouTube smoke: YOUTUBE_API_KEY is not set")
        return 0

    snapshot = await run_youtube_search(
        _search_plan(),
        api_key=api_key,
        max_results_per_intent=3,
        top_n=3,
        llm_candidate_limit=3,
    )
    assert snapshot.raw_candidate_count >= 0
    assert snapshot.quota_estimate.estimated_units >= 101
    if snapshot.raw_candidate_count:
        assert snapshot.ranked_candidates, [warning.model_dump() for warning in snapshot.warnings]
        assert all(candidate.llm_score is not None for candidate in snapshot.ranked_candidates)

    with tempfile.TemporaryDirectory(prefix="knowledge-loop-youtube-smoke-") as tmp:
        paths = persist_youtube_search_snapshot(snapshot, state_root=Path(tmp))
        assert len(paths) == 2

    print("youtube search smoke completed")
    for candidate in snapshot.ranked_candidates:
        print(f"{candidate.final_score:.4f} {candidate.title}")
    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())

