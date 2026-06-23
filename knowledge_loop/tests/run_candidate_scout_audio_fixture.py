from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from knowledge_loop.candidate_scout import (
    persist_candidate_scout_snapshot,
    run_candidate_audio_extraction,
)
from knowledge_loop.models import (
    CandidateAudioSegment,
    CandidateDownloadManifest,
    CandidateDownloadRecord,
    KnowledgeGap,
)


def _gap() -> KnowledgeGap:
    return KnowledgeGap(
        gap_id="gap-scout-audio-fixture",
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
        final_score=0.9 - idx * 0.1,
        llm_score=0.9 - idx * 0.1,
        metadata_score=0.8 - idx * 0.1,
        download_dir=str(download_dir),
        local_video_path=str(video_path),
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        video_one = root / "candidate_one.mp4"
        video_two = root / "candidate_two.mp4"
        video_one.write_bytes(b"fake video one")
        video_two.write_bytes(b"fake video two")

        manifest = CandidateDownloadManifest(
            generated_at=datetime(2026, 5, 13, tzinfo=timezone.utc),
            loop_profile_id="league_of_legends",
            target_gap=_gap(),
            requested_top_n=3,
            selected_candidate_count=2,
            successful_download_count=2,
            download_root=str(root),
            records=[
                _download_record(1, video_one, root / "download_one"),
                _download_record(2, video_two, root / "download_two"),
            ],
        )

        def fake_audio_extractor(
            video_path: Path,
            working_dir: Path,
            segment_length: int,
            audio_output_format: str,
        ) -> list[CandidateAudioSegment]:
            cache_dir = working_dir / "_cache" / video_path.stem
            cache_dir.mkdir(parents=True, exist_ok=True)
            segments = []
            for idx in range(2):
                segment_name = f"fixture-{idx}-{idx * segment_length}-{(idx + 1) * segment_length}"
                audio_path = cache_dir / f"{segment_name}.{audio_output_format}"
                audio_path.write_bytes(b"fake audio")
                segments.append(
                    CandidateAudioSegment(
                        segment_index=idx,
                        segment_name=segment_name,
                        start_seconds=idx * segment_length,
                        end_seconds=(idx + 1) * segment_length,
                        audio_path=str(audio_path),
                        audio_exists=True,
                    )
                )
            return segments

        snapshot = run_candidate_audio_extraction(
            manifest,
            download_manifest_path="/tmp/latest_download_manifest.json",
            top_n=3,
            audio_extractor=fake_audio_extractor,
        )
        assert snapshot.selected_candidate_count == 2
        assert snapshot.audio_extracted_count == 2
        assert snapshot.failed_count == 0
        assert len(snapshot.records) == 2
        assert snapshot.records[0].candidate_id == "cand-scout-1"
        assert snapshot.records[0].status == "audio_extracted"
        assert snapshot.records[0].audio_segment_count == 2
        assert Path(snapshot.records[0].audio_segments[0].audio_path).exists()
        assert len([w for w in snapshot.warnings if w.code == "insufficient_downloaded_candidates"]) == 1

        paths = persist_candidate_scout_snapshot(snapshot, state_root=root)
        assert len(paths) == 2
        for path in paths:
            assert path.exists()

    print("candidate scout audio fixture test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
