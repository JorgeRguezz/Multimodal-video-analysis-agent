from __future__ import annotations

from datetime import datetime, timezone

from knowledge_loop.models import KnowledgeGap, SearchIntent, SearchPlanSnapshot
from knowledge_loop.youtube_search import (
    _parse_llm_candidate_scores,
    candidates_from_video_items,
    collect_video_ids_from_search_results,
    parse_youtube_duration,
    rerank_candidates_with_llm,
    score_candidates_with_metadata,
)


def _search_plan() -> SearchPlanSnapshot:
    return SearchPlanSnapshot(
        generated_at=datetime.now(timezone.utc),
        loop_profile_id="league_of_legends",
        target_gap=KnowledgeGap(
            gap_id="gap-youtube-fixture",
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
            evidence_summary="RUNES::AATROX / runes is weakly covered.",
            reasons=["low_coverage", "weak_redundancy", "freshness_unknown"],
        ),
        search_intents=[
            SearchIntent(
                query="Aatrox rune guide current season",
                purpose="Find Aatrox rune setup explanations.",
                source_type="educational_guide",
                expected_signal="Explains keystone and rune choices.",
            ),
            SearchIntent(
                query="Aatrox best runes explained",
                purpose="Find rune recommendations with reasoning.",
                source_type="current_patch_or_season_guide",
            ),
        ],
    )


def main() -> int:
    assert parse_youtube_duration("PT12M34S") == 754
    assert parse_youtube_duration("PT1H2M3S") == 3723
    assert parse_youtube_duration("bad") is None

    plan = _search_plan()
    search_payloads = [
        (
            plan.search_intents[0],
            {
                "items": [
                    {"id": {"videoId": "video-a"}},
                    {"id": {"videoId": "video-b"}},
                ]
            },
        ),
        (
            plan.search_intents[1],
            {
                "items": [
                    {"id": {"videoId": "video-a"}},
                ]
            },
        ),
    ]
    matched = collect_video_ids_from_search_results(search_payloads)
    assert matched["video-a"] == {
        "Aatrox rune guide current season",
        "Aatrox best runes explained",
    }
    assert matched["video-b"] == {"Aatrox rune guide current season"}

    video_items = [
        {
            "id": "video-a",
            "snippet": {
                "title": "Aatrox Rune Guide Current Season",
                "description": "A detailed Aatrox rune guide explaining Conqueror and secondary runes.",
                "channelId": "channel-a",
                "channelTitle": "Top Lane Coach",
                "publishedAt": "2026-03-01T12:00:00Z",
            },
            "contentDetails": {"duration": "PT12M10S"},
            "statistics": {"viewCount": "12000", "likeCount": "900", "commentCount": "99"},
        },
        {
            "id": "video-b",
            "snippet": {
                "title": "Funny Aatrox Montage",
                "description": "Clips and highlights.",
                "channelId": "channel-b",
                "channelTitle": "Montage Hub",
                "publishedAt": "2022-01-01T12:00:00Z",
            },
            "contentDetails": {"duration": "PT2M10S"},
            "statistics": {"viewCount": "999999", "likeCount": "1000"},
        },
    ]
    candidates = candidates_from_video_items(video_items, matched)
    assert len(candidates) == 2
    assert candidates[0].duration_seconds == 730
    scored = score_candidates_with_metadata(
        candidates,
        plan,
        now=datetime(2026, 5, 4, tzinfo=timezone.utc),
    )
    assert scored[0].youtube_video_id == "video-a"
    assert scored[0].metadata_components["duration_fit_score"] == 1.0
    assert "title_matches_gap" in scored[0].ranking_reasons

    llm_json = {
        "ranked_candidates": [
            {
                "candidate_id": scored[0].candidate_id,
                "llm_score": 0.92,
                "reason": "Directly targets Aatrox runes and appears educational.",
            },
            {
                "candidate_id": scored[1].candidate_id,
                "llm_score": 0.12,
                "reason": "Montage format is unlikely to explain rune decisions.",
            },
        ]
    }
    parsed, warnings = _parse_llm_candidate_scores(__import__("json").dumps(llm_json))
    assert not warnings
    assert parsed[scored[0].candidate_id][0] == 0.92

    wrapped_llm_output = (
        "<|start|>assistant<|channel|>analysis<|message|>"
        "I will compare the candidates first."
        "<|end|>\n"
        "<|start|>assistant<|channel|>final<|message|>"
        "```json\n"
        + __import__("json").dumps(llm_json)
        + "\n```"
        "<|return|>"
    )
    parsed, warnings = _parse_llm_candidate_scores(wrapped_llm_output)
    assert not warnings
    assert parsed[scored[1].candidate_id][0] == 0.12

    print("youtube search fixture test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
