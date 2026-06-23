from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from knowledge_loop.candidate_scout import (
    build_selection_snapshot,
    persist_candidate_scout_snapshot,
    persist_candidate_selection_snapshot,
    run_full_candidate_scout,
)
from knowledge_loop.models import (
    CandidateAudioSegment,
    CandidateDownloadManifest,
    CandidateDownloadRecord,
    InventorySnapshot,
    InventorySourceRef,
    KnowledgeGap,
    TopicFacetInventory,
    VideoInventory,
)


def _gap() -> KnowledgeGap:
    return KnowledgeGap(
        gap_id="gap-scout-fixture",
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


def _download_record(idx: int, video_path: Path, download_dir: Path) -> CandidateDownloadRecord:
    return CandidateDownloadRecord(
        candidate_id=f"cand-scout-{idx}",
        rank=idx,
        status="downloaded",
        source_url=f"https://www.youtube.com/watch?v=scout-{idx}",
        youtube_video_id=f"scout-{idx}",
        title=f"Darius Scout Candidate {idx}",
        duration_seconds=600,
        final_score=0.9 - idx * 0.1,
        llm_score=0.9 - idx * 0.1,
        metadata_score=0.8 - idx * 0.1,
        download_dir=str(download_dir),
        local_video_path=str(video_path),
    )


def _inventory(root: Path) -> InventorySnapshot:
    return InventorySnapshot(
        generated_at=datetime(2026, 5, 13, tzinfo=timezone.utc),
        loop_profile_id="league_of_legends",
        extraction_game_id="league_of_legends",
        source_root=str(root),
        topic_facets=[
            TopicFacetInventory(
                topic_key="PATHING::DARIUS",
                facet="pathing",
                video_count=1,
                chunk_count=1,
                graph_entity_count=0,
                source_redundancy=1,
                source_diversity_score=0.2,
                weak_redundancy_score=0.8,
                coverage_score=0.2,
                supporting_videos=["legacy"],
                evidence=[
                    InventorySourceRef(
                        source_kind="chunk",
                        video_name="legacy",
                        text_preview="Darius can path through river after pushing top wave.",
                    )
                ],
            )
        ],
    )


async def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        records = []
        for idx in range(1, 4):
            video_path = root / f"candidate_{idx}.mp4"
            video_path.write_bytes(f"fake video {idx}".encode("utf-8"))
            records.append(_download_record(idx, video_path, root / f"download_{idx}"))
        manifest = CandidateDownloadManifest(
            generated_at=datetime(2026, 5, 13, tzinfo=timezone.utc),
            loop_profile_id="league_of_legends",
            target_gap=_gap(),
            requested_top_n=3,
            selected_candidate_count=3,
            successful_download_count=3,
            download_root=str(root),
            records=records,
        )

        inventory = _inventory(root)
        inventory.videos["known-duplicate"] = VideoInventory(
            video_name="known-duplicate",
            cache_dir=str(root / "known"),
            source_path="https://www.youtube.com/watch?v=scout-3",
        )

        def fake_audio_extractor(video_path: Path, working_dir: Path, segment_length: int, audio_output_format: str):
            cache_dir = working_dir / "_cache" / video_path.stem
            cache_dir.mkdir(parents=True, exist_ok=True)
            audio_path = cache_dir / f"fixture-0-0-{segment_length}.{audio_output_format}"
            audio_path.write_bytes(b"fake audio")
            return [
                CandidateAudioSegment(
                    segment_index=0,
                    segment_name=f"fixture-0-0-{segment_length}",
                    start_seconds=0,
                    end_seconds=segment_length,
                    audio_path=str(audio_path),
                    audio_exists=True,
                )
            ]

        def fake_transcript_provider(record):
            return (
                f"{record.title}. This transcript explains Darius pathing, wave movement, "
                "river routes, and when to rotate after pushing lane."
            )

        async def fake_embedding_func(texts: list[str]):
            rows = []
            for text in texts:
                lower = text.lower()
                rows.append(
                    [
                        1.0 if "darius" in lower else 0.0,
                        1.0 if "path" in lower else 0.0,
                        1.0 if "wave" in lower else 0.0,
                        min(1.0, len(lower) / 500.0),
                    ]
                )
            return rows

        def fake_llm_reviewer(record, transcript):
            scores = {
                "cand-scout-1": 0.92,
                "cand-scout-2": 0.86,
                "cand-scout-3": 0.95,
            }
            return {
                "candidate_id": record.candidate_id,
                "relevance_score": scores[record.candidate_id],
                "decision": "accept",
                "reason": "Transcript directly discusses Darius pathing decisions.",
                "supporting_evidence": ["explains Darius pathing and wave movement"],
                "risks": [],
            }

        scout = await run_full_candidate_scout(
            manifest,
            download_manifest_path="/tmp/latest_download_manifest.json",
            inventory=inventory,
            top_n=3,
            audio_extractor=fake_audio_extractor,
            transcript_provider=fake_transcript_provider,
            embedding_func=fake_embedding_func,
            llm_reviewer=fake_llm_reviewer,
            acceptance_threshold=0.85,
        )
        assert scout.selected_candidate_count == 3
        assert scout.failed_count == 0
        assert all(record.status == "reviewed" for record in scout.records)
        assert [record.accepted_for_queue for record in scout.records] == [True, True, False]
        assert scout.records[2].source_duplicate_score == 1.0
        assert scout.records[0].transcript_relevance_score is not None
        assert scout.records[0].content_redundancy_score is not None

        selection = build_selection_snapshot(scout, scout_snapshot_path="/tmp/latest_candidate_scout.json")
        assert selection.accepted_count == 2
        assert selection.rejected_count == 1
        assert [record.candidate_id for record in selection.records if record.accepted_for_queue] == [
            "cand-scout-1",
            "cand-scout-2",
        ]

        scout_paths = persist_candidate_scout_snapshot(scout, state_root=root)
        selection_paths = persist_candidate_selection_snapshot(selection, state_root=root)
        for path in scout_paths + selection_paths:
            assert path.exists()

    print("candidate scout fixture test passed")
    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main()))
