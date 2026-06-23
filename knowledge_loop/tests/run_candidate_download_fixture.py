from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from knowledge_loop.candidate_download import (
    DownloadOutcome,
    persist_download_manifest,
    run_candidate_downloads,
    select_download_candidates,
)
from knowledge_loop.models import KnowledgeGap, SearchIntent, YouTubeCandidate, YouTubeSearchRunSnapshot


def _gap() -> KnowledgeGap:
    return KnowledgeGap(
        gap_id="gap-download-fixture",
        gap_kind="low_coverage",
        topic_key="PATHING::DARIUS",
        facet="pathing",
        priority=0.50,
        status="open",
        coverage_score=0.30,
        coverage_gap_score=0.70,
        missing_facet_score=0.0,
        freshness_days=None,
        freshness_score=None,
        weak_redundancy_score=0.60,
        actionability_score=0.90,
        video_count=2,
        chunk_count=3,
        source_redundancy=2,
        graph_entity_count=1,
        evidence_summary="PATHING::DARIUS / pathing is weakly covered.",
        reasons=["low_coverage", "weak_redundancy"],
    )


def _candidate(idx: int, *, score: float, url: str | None = None) -> YouTubeCandidate:
    return YouTubeCandidate(
        candidate_id=f"cand-fixture-{idx}",
        youtube_video_id=f"video-{idx}",
        source_url=url or f"https://www.youtube.com/watch?v=video-{idx}",
        title=f"Darius Pathing Candidate {idx}",
        description="Educational Darius pathing guide.",
        channel_title="Fixture Channel",
        publish_date=datetime(2026, 5, idx, tzinfo=timezone.utc),
        duration_seconds=600 + idx,
        view_count=1000 * idx,
        like_count=50 * idx,
        matched_intents=["Darius pathing guide"],
        metadata_score=score - 0.05,
        llm_score=score,
        final_score=score,
    )


def _search_run() -> YouTubeSearchRunSnapshot:
    return YouTubeSearchRunSnapshot(
        generated_at=datetime(2026, 5, 5, tzinfo=timezone.utc),
        loop_profile_id="league_of_legends",
        search_plan_path="/tmp/search_plan.json",
        target_gap=_gap(),
        search_intents=[
            SearchIntent(
                query="Darius pathing guide",
                purpose="Find pathing videos.",
                source_type="educational_guide",
            )
        ],
        raw_candidate_count=5,
        llm_ranked_candidate_count=5,
        candidate_count=5,
        ranked_candidates=[
            _candidate(1, score=0.90),
            _candidate(2, score=0.80),
            _candidate(3, score=0.70),
            _candidate(4, score=0.60),
            _candidate(5, score=0.50),
        ],
    )


def main() -> int:
    snapshot = _search_run()
    selected, warnings = select_download_candidates(snapshot, top_n=3)
    assert not warnings
    assert [candidate.candidate_id for candidate in selected] == [
        "cand-fixture-1",
        "cand-fixture-2",
        "cand-fixture-3",
    ]

    def fake_downloader(candidate: YouTubeCandidate, download_dir: Path) -> DownloadOutcome:
        if candidate.candidate_id == "cand-fixture-2":
            raise RuntimeError("fixture download failure")
        download_dir.mkdir(parents=True, exist_ok=True)
        video_path = download_dir / "video.mp4"
        info_path = download_dir / "yt_dlp_info.json"
        video_path.write_bytes(b"fixture video")
        info_path.write_text('{"fixture": true}', encoding="utf-8")
        return DownloadOutcome(local_video_path=video_path, yt_dlp_info_path=info_path, cookie_source="fixture")

    with tempfile.TemporaryDirectory() as tmp:
        manifest = run_candidate_downloads(
            snapshot,
            search_run_path="/tmp/latest_youtube_search.json",
            state_root=tmp,
            top_n=3,
            downloader=fake_downloader,
        )
        assert manifest.selected_candidate_count == 3
        assert manifest.successful_download_count == 2
        assert manifest.failed_download_count == 1
        assert manifest.records[0].status == "downloaded"
        assert manifest.records[0].cookie_source == "fixture"
        assert manifest.records[1].status == "failed"
        assert manifest.records[1].failure_reason == "fixture download failure"
        assert manifest.records[2].status == "downloaded"
        assert Path(manifest.records[0].local_video_path).exists()
        assert Path(manifest.records[2].local_video_path).exists()
        assert len([w for w in manifest.warnings if w.code == "candidate_download_failed"]) == 1

        paths = persist_download_manifest(manifest, state_root=tmp)
        assert len(paths) == 2
        for path in paths:
            assert path.exists()

    print("candidate download fixture test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
