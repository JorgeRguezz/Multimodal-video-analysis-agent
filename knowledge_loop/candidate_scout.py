from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import config
from .models import (
    CandidateAudioSegment,
    CandidateDownloadManifest,
    CandidateDownloadRecord,
    CandidateScoutRecord,
    CandidateScoutSnapshot,
    CandidateScoutWarning,
    CandidateSelectionRecord,
    CandidateSelectionSnapshot,
    InventorySnapshot,
)
from .prompts import SYSTEM_CANDIDATE_TRANSCRIPT_REVIEWER, USER_CANDIDATE_TRANSCRIPT_REVIEW_TEMPLATE
from .search_plan import _extract_json_object


DEFAULT_SCOUT_TOP_N = 3
DEFAULT_SCOUT_SEGMENT_LENGTH = 30
DEFAULT_SCOUT_FRAMES_PER_SEGMENT = 1
DEFAULT_SCOUT_AUDIO_FORMAT = "mp3"
DEFAULT_ACCEPTANCE_THRESHOLD = 0.85
DEFAULT_TRANSCRIPT_CHUNK_CHARS = 1200
DEFAULT_REVIEW_TRANSCRIPT_CHAR_LIMIT = 60000


AudioExtractor = Callable[[Path, Path, int, str], list[CandidateAudioSegment]]
TranscriptProvider = Callable[[CandidateScoutRecord], str | Awaitable[str]]
LLMReviewer = Callable[[CandidateScoutRecord, str], dict[str, Any] | Awaitable[dict[str, Any]]]
EmbeddingFunc = Callable[[list[str]], Any]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: dict[str, Any], pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2 if pretty else None, ensure_ascii=False)


def load_download_manifest(path: str | Path) -> CandidateDownloadManifest:
    return CandidateDownloadManifest.model_validate(_read_json(Path(path)))


def load_inventory_snapshot(path: str | Path | None) -> InventorySnapshot | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    return InventorySnapshot.model_validate(_read_json(resolved))


def _build_audio_segments(
    *,
    video_path: Path,
    working_dir: Path,
    segment_index2name: dict[Any, Any],
    segment_times_info: dict[Any, Any],
    audio_output_format: str,
) -> list[CandidateAudioSegment]:
    video_name = video_path.stem
    cache_dir = working_dir / "_cache" / video_name
    segments: list[CandidateAudioSegment] = []
    for raw_index, raw_name in sorted(segment_index2name.items(), key=lambda item: int(item[0])):
        index = int(raw_index)
        segment_name = str(raw_name)
        timestamp = segment_times_info.get(str(raw_index), {}).get("timestamp")
        if timestamp is None:
            timestamp = segment_times_info.get(raw_index, {}).get("timestamp", (0, 0))
        start_seconds, end_seconds = float(timestamp[0]), float(timestamp[1])
        audio_path = cache_dir / f"{segment_name}.{audio_output_format}"
        segments.append(
            CandidateAudioSegment(
                segment_index=index,
                segment_name=segment_name,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                audio_path=str(audio_path.resolve()) if audio_path.exists() else None,
                audio_exists=audio_path.exists(),
            )
        )
    return segments


def extract_audio_segments_like_knowledge_extraction(
    video_path: Path,
    working_dir: Path,
    segment_length: int = DEFAULT_SCOUT_SEGMENT_LENGTH,
    audio_output_format: str = DEFAULT_SCOUT_AUDIO_FORMAT,
) -> list[CandidateAudioSegment]:
    """Extract 30s segment MP3s using the same splitter used by knowledge_extraction."""
    from knowledge_build._videoutil.split import split_video  # noqa: WPS433

    working_dir.mkdir(parents=True, exist_ok=True)
    segment_index2name, segment_times_info = split_video(
        video_path=str(video_path),
        working_dir=str(working_dir),
        segment_length=segment_length,
        num_frames_per_segment=DEFAULT_SCOUT_FRAMES_PER_SEGMENT,
        audio_output_format=audio_output_format,
    )
    return _build_audio_segments(
        video_path=video_path,
        working_dir=working_dir,
        segment_index2name=segment_index2name,
        segment_times_info=segment_times_info,
        audio_output_format=audio_output_format,
    )


def _record_base(download_record: CandidateDownloadRecord) -> dict[str, Any]:
    return {
        "candidate_id": download_record.candidate_id,
        "rank": download_record.rank,
        "source_url": download_record.source_url,
        "youtube_video_id": download_record.youtube_video_id,
        "title": download_record.title,
        "duration_seconds": download_record.duration_seconds,
        "local_video_path": download_record.local_video_path,
        "download_dir": download_record.download_dir,
    }


def _candidate_working_dir(download_record: CandidateDownloadRecord) -> Path:
    return Path(download_record.download_dir) / "scout_audio"


def run_candidate_audio_extraction(
    manifest: CandidateDownloadManifest,
    *,
    download_manifest_path: str | None = None,
    top_n: int = DEFAULT_SCOUT_TOP_N,
    audio_extractor: AudioExtractor | None = None,
    segment_length: int = DEFAULT_SCOUT_SEGMENT_LENGTH,
    audio_output_format: str = DEFAULT_SCOUT_AUDIO_FORMAT,
    dry_run: bool = False,
) -> CandidateScoutSnapshot:
    extractor = audio_extractor or extract_audio_segments_like_knowledge_extraction
    generated_at = datetime.now(timezone.utc)
    records: list[CandidateScoutRecord] = []
    warnings: list[CandidateScoutWarning] = []
    eligible = [record for record in manifest.records if record.status == "downloaded"][: max(1, top_n)]

    for download_record in eligible:
        base = _record_base(download_record)
        working_dir = _candidate_working_dir(download_record)
        if dry_run:
            records.append(
                CandidateScoutRecord(
                    **base,
                    status="skipped",
                    audio_working_dir=str(working_dir),
                    failure_reason="dry_run",
                )
            )
            continue

        if not download_record.local_video_path:
            message = "Downloaded candidate record has no local_video_path"
            warnings.append(
                CandidateScoutWarning(
                    code="missing_local_video_path",
                    message=message,
                    candidate_id=download_record.candidate_id,
                )
            )
            records.append(
                CandidateScoutRecord(
                    **base,
                    status="failed",
                    audio_working_dir=str(working_dir),
                    failure_reason=message,
                )
            )
            continue

        video_path = Path(download_record.local_video_path)
        if not video_path.exists():
            message = f"Downloaded video file does not exist: {video_path}"
            warnings.append(
                CandidateScoutWarning(
                    code="local_video_missing",
                    message=message,
                    candidate_id=download_record.candidate_id,
                )
            )
            records.append(
                CandidateScoutRecord(
                    **base,
                    status="failed",
                    audio_working_dir=str(working_dir),
                    failure_reason=message,
                )
            )
            continue

        try:
            segments = extractor(video_path, working_dir, segment_length, audio_output_format)
            existing_segments = [segment for segment in segments if segment.audio_exists]
            records.append(
                CandidateScoutRecord(
                    **base,
                    status="audio_extracted",
                    audio_working_dir=str(working_dir),
                    audio_segments=segments,
                    audio_segment_count=len(existing_segments),
                    extracted_at=datetime.now(timezone.utc),
                )
            )
            if not existing_segments:
                warnings.append(
                    CandidateScoutWarning(
                        code="no_audio_segments_extracted",
                        message="Audio extraction completed but produced no segment audio files",
                        candidate_id=download_record.candidate_id,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            warnings.append(
                CandidateScoutWarning(
                    code="audio_extraction_failed",
                    message=message,
                    candidate_id=download_record.candidate_id,
                )
            )
            records.append(
                CandidateScoutRecord(
                    **base,
                    status="failed",
                    audio_working_dir=str(working_dir),
                    failure_reason=message,
                )
            )

    if len(eligible) < max(1, top_n):
        warnings.append(
            CandidateScoutWarning(
                code="insufficient_downloaded_candidates",
                message=f"Found {len(eligible)} downloaded candidates; requested {max(1, top_n)}",
            )
        )

    return CandidateScoutSnapshot(
        generated_at=generated_at,
        loop_profile_id=manifest.loop_profile_id,
        download_manifest_path=download_manifest_path,
        download_manifest_generated_at=manifest.generated_at,
        target_gap=manifest.target_gap,
        requested_top_n=max(1, top_n),
        selected_candidate_count=len(eligible),
        audio_extracted_count=sum(1 for record in records if record.status == "audio_extracted"),
        failed_count=sum(1 for record in records if record.status == "failed"),
        skipped_count=sum(1 for record in records if record.status == "skipped"),
        records=records,
        warnings=warnings,
    )


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _chunk_text(text: str, max_chars: int = DEFAULT_TRANSCRIPT_CHUNK_CHARS) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs or [text]:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for idx in range(0, len(paragraph), max_chars):
                chunks.append(paragraph[idx : idx + max_chars].strip())
            continue
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current.strip())
            current = paragraph
    if current:
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_'-]+", str(text or "").lower()))


def _target_query(record: CandidateScoutRecord, snapshot: CandidateScoutSnapshot) -> str:
    gap = snapshot.target_gap
    return _clean_text(
        " ".join(
            [
                gap.topic_key.replace("::", " "),
                gap.facet,
                gap.evidence_summary,
                " ".join(gap.reasons),
                record.title,
            ]
        )
    )


async def transcribe_audio_segments_with_existing_asr(record: CandidateScoutRecord) -> str:
    """Use the same MCP ASR server used by knowledge_extraction.extractor."""
    from knowledge_extraction.config import VENV_VLM_ASR_PYTHON  # noqa: WPS433
    from mcp import ClientSession, StdioServerParameters  # noqa: WPS433
    from mcp.client.stdio import stdio_client  # noqa: WPS433

    audio_paths = [
        segment.audio_path
        for segment in record.audio_segments
        if segment.audio_exists and segment.audio_path and Path(segment.audio_path).exists()
    ]
    if not audio_paths:
        return ""

    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    params = StdioServerParameters(
        command=VENV_VLM_ASR_PYTHON,
        args=["-m", "knowledge_extraction.vlm_asr_server"],
        env=env,
    )
    transcript_parts: list[str] = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for segment in record.audio_segments:
                if not segment.audio_exists or not segment.audio_path or not Path(segment.audio_path).exists():
                    continue
                result = await session.call_tool("transcribe_audio", arguments={"audio_path": segment.audio_path})
                text = result.content[0].text if result.content else ""
                if text:
                    transcript_parts.append(
                        f"[segment {segment.segment_index} {segment.start_seconds:.2f}s-{segment.end_seconds:.2f}s]\n{text}"
                    )
            try:
                await session.call_tool("unload_vlm_asr", arguments={})
            except Exception:
                pass
    return "\n\n".join(transcript_parts).strip()


def _write_transcript(record: CandidateScoutRecord, transcript: str) -> str:
    base_dir = Path(record.download_dir or record.audio_working_dir or ".")
    path = base_dir / "scout_transcript.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(transcript, encoding="utf-8")
    return str(path.resolve())


async def _embed_texts(texts: list[str], embedding_func: EmbeddingFunc | None = None) -> list[list[float]]:
    if not texts:
        return []
    if embedding_func is None:
        from knowledge_build._llm import local_llm_config  # noqa: WPS433

        embedding_func = local_llm_config.embedding_func
    raw = await _maybe_await(embedding_func(texts))
    return [list(map(float, row)) for row in raw]


def _cosine(a: list[float], b: list[float]) -> float:
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if denom <= 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / denom


def _normalize_similarity(value: float) -> float:
    return max(0.0, min(1.0, (value + 1.0) / 2.0))


async def compute_transcript_relevance_score(
    record: CandidateScoutRecord,
    snapshot: CandidateScoutSnapshot,
    transcript: str,
    *,
    embedding_func: EmbeddingFunc | None = None,
) -> float | None:
    chunks = _chunk_text(transcript)
    if not chunks:
        return None
    query = _target_query(record, snapshot)
    embeddings = await _embed_texts([query, *chunks], embedding_func=embedding_func)
    if len(embeddings) < 2:
        return None
    query_embedding = embeddings[0]
    similarities = sorted(
        (_normalize_similarity(_cosine(query_embedding, chunk_embedding)) for chunk_embedding in embeddings[1:]),
        reverse=True,
    )
    top_3 = similarities[:3]
    embedding_score = 0.70 * similarities[0] + 0.30 * (sum(top_3) / len(top_3))
    target_tokens = _tokens(query)
    transcript_tokens = _tokens(transcript)
    exact_signal = len(target_tokens & transcript_tokens) / max(1, min(len(target_tokens), 12))
    return round(max(0.0, min(1.0, 0.85 * embedding_score + 0.15 * min(1.0, exact_signal))), 4)


def _youtube_id_from_url(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urllib.parse.urlparse(str(value))
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        return parsed.path.strip("/") or None
    if "youtube.com" in host:
        query_id = urllib.parse.parse_qs(parsed.query).get("v", [None])[0]
        if query_id:
            return query_id
        match = re.search(r"/(?:shorts|embed)/([^/?#]+)", parsed.path)
        if match:
            return match.group(1)
    return None


def _canonical_url(value: str | None) -> str | None:
    if not value:
        return None
    youtube_id = _youtube_id_from_url(value)
    if youtube_id:
        return f"youtube:{youtube_id}"
    parsed = urllib.parse.urlparse(str(value).strip())
    if not parsed.scheme or not parsed.netloc:
        return str(value).strip().lower() or None
    return urllib.parse.urlunparse(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", parsed.query, "")
    ).lower()


def _source_ids_from_download_record(record: CandidateDownloadRecord | CandidateScoutRecord) -> set[str]:
    ids: set[str] = set()
    youtube_video_id = getattr(record, "youtube_video_id", None)
    source_url = getattr(record, "source_url", None)
    if youtube_video_id:
        ids.add(f"youtube:{youtube_video_id}")
    canonical = _canonical_url(source_url)
    if canonical:
        ids.add(canonical)
    info_path = getattr(record, "yt_dlp_info_path", None)
    if info_path and Path(info_path).exists():
        try:
            info = _read_json(Path(info_path))
        except Exception:
            info = {}
        for key in ("id", "display_id"):
            if info.get(key):
                ids.add(f"youtube:{info[key]}")
        for key in ("webpage_url", "original_url", "url"):
            canonical_info = _canonical_url(str(info.get(key) or ""))
            if canonical_info:
                ids.add(canonical_info)
    return {item.lower() for item in ids if item}


def _source_ids_from_inventory(inventory: InventorySnapshot | None) -> set[str]:
    if inventory is None:
        return set()
    ids: set[str] = set()
    for video in inventory.videos.values():
        canonical = _canonical_url(video.source_path)
        if canonical:
            ids.add(canonical.lower())
    return ids


def compute_source_duplicate_score(
    record: CandidateScoutRecord,
    manifest: CandidateDownloadManifest,
    inventory: InventorySnapshot | None = None,
) -> tuple[float, str]:
    candidate_ids = _source_ids_from_download_record(record)
    if not candidate_ids:
        return 0.0, "no robust source identifiers available"
    known_ids = _source_ids_from_inventory(inventory)
    for other in manifest.records:
        if other.candidate_id == record.candidate_id:
            continue
        known_ids.update(_source_ids_from_download_record(other))
    overlap = candidate_ids & known_ids
    if overlap:
        return 1.0, f"exact source identifier match: {sorted(overlap)[0]}"
    return 0.0, "no exact source identifier match"


def _inventory_evidence_texts(snapshot: CandidateScoutSnapshot, inventory: InventorySnapshot | None) -> list[str]:
    if inventory is None:
        return []
    gap = snapshot.target_gap
    texts: list[str] = []
    for bucket in inventory.topic_facets:
        if bucket.topic_key != gap.topic_key or bucket.facet != gap.facet:
            continue
        for evidence in bucket.evidence:
            if evidence.text_preview:
                texts.append(evidence.text_preview)
    return texts


async def compute_content_redundancy_score(
    snapshot: CandidateScoutSnapshot,
    transcript: str,
    inventory: InventorySnapshot | None,
    *,
    embedding_func: EmbeddingFunc | None = None,
) -> tuple[float | None, str | None]:
    evidence_texts = _inventory_evidence_texts(snapshot, inventory)
    transcript_chunks = _chunk_text(transcript)
    if not evidence_texts:
        return None, "no same-topic inventory evidence available"
    if not transcript_chunks:
        return None, "no transcript chunks available"
    transcript_sample = transcript_chunks[:12]
    evidence_sample = evidence_texts[:24]
    embeddings = await _embed_texts([*transcript_sample, *evidence_sample], embedding_func=embedding_func)
    if len(embeddings) < len(transcript_sample) + 1:
        return None, "embedding unavailable"
    transcript_embeddings = embeddings[: len(transcript_sample)]
    evidence_embeddings = embeddings[len(transcript_sample) :]
    best_scores = []
    for transcript_embedding in transcript_embeddings:
        best_scores.append(
            max((_normalize_similarity(_cosine(transcript_embedding, evidence_embedding)) for evidence_embedding in evidence_embeddings), default=0.0)
        )
    if not best_scores:
        return None, "no comparable transcript/evidence embeddings"
    top_3 = sorted(best_scores, reverse=True)[:3]
    score = 0.70 * max(best_scores) + 0.30 * (sum(top_3) / len(top_3))
    return round(max(0.0, min(1.0, score)), 4), "embedding similarity against same-topic inventory evidence"


def compute_extractability_score(record: CandidateScoutRecord, transcript: str) -> float:
    transcript_len_score = min(1.0, len(transcript.strip()) / 2500)
    audio_score = min(1.0, record.audio_segment_count / 2)
    duration_score = 0.5
    if record.duration_seconds:
        if 180 <= record.duration_seconds <= 2400:
            duration_score = 1.0
        elif record.duration_seconds < 180:
            duration_score = max(0.2, record.duration_seconds / 180)
        else:
            duration_score = max(0.25, 1.0 - min(0.75, (record.duration_seconds - 2400) / 3600))
    return round(max(0.0, min(1.0, 0.50 * transcript_len_score + 0.30 * audio_score + 0.20 * duration_score)), 4)


def _postprocess_llm_json_output(text: str) -> str:
    cleaned = str(text or "").strip()
    final_marker = "<|channel|>final<|message|>"
    idx = cleaned.rfind(final_marker)
    if idx >= 0:
        cleaned = cleaned[idx + len(final_marker) :]
    for marker in ("<|return|>", "<|end|>", "<|start|>"):
        end_idx = cleaned.find(marker)
        if end_idx >= 0:
            cleaned = cleaned[:end_idx]
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned.strip(), flags=re.IGNORECASE | re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    if "{" in cleaned:
        cleaned = cleaned[cleaned.find("{") :]
    return cleaned.strip()


def _review_transcript_for_prompt(transcript: str, limit: int = DEFAULT_REVIEW_TRANSCRIPT_CHAR_LIMIT) -> tuple[str, bool]:
    if limit <= 0 or len(transcript) <= limit:
        return transcript, False
    return transcript[:limit].rstrip() + "\n\n[Transcript truncated for context budget.]", True


def _format_review_prompt(
    record: CandidateScoutRecord,
    snapshot: CandidateScoutSnapshot,
    transcript: str,
) -> tuple[str, bool]:
    prompt_transcript, truncated = _review_transcript_for_prompt(transcript)
    gap = snapshot.target_gap
    prompt = USER_CANDIDATE_TRANSCRIPT_REVIEW_TEMPLATE.format(
        gap_id=gap.gap_id,
        gap_kind=gap.gap_kind,
        topic_key=gap.topic_key,
        facet=gap.facet,
        priority=gap.priority,
        evidence_summary=gap.evidence_summary,
        gap_reasons=", ".join(gap.reasons) if gap.reasons else "none",
        candidate_id=record.candidate_id,
        rank=record.rank,
        title=record.title,
        source_url=record.source_url,
        youtube_video_id=record.youtube_video_id,
        duration_seconds=record.duration_seconds,
        first_pass_final_score="unknown",
        transcript_relevance_score=record.transcript_relevance_score,
        source_duplicate_score=record.source_duplicate_score,
        source_duplicate_reason=record.source_duplicate_reason,
        content_redundancy_score=record.content_redundancy_score,
        content_redundancy_reason=record.content_redundancy_reason,
        extractability_score=record.extractability_score,
        transcript_char_count=len(transcript),
        transcript_truncated_for_review=truncated,
        transcript=prompt_transcript,
    )
    return prompt, truncated


async def review_candidate_with_llm(record: CandidateScoutRecord, snapshot: CandidateScoutSnapshot, transcript: str) -> dict[str, Any]:
    from knowledge_build._llm import local_llm_config  # noqa: WPS433

    prompt, truncated = _format_review_prompt(record, snapshot, transcript)
    result = await local_llm_config.best_model_func(
        prompt,
        system_prompt=SYSTEM_CANDIDATE_TRANSCRIPT_REVIEWER,
        max_tokens=2000,
        temperature=0.1,
        top_p=1.0,
        top_k=0,
        repeat_penalty=1.08,
        return_metadata=True,
    )
    if isinstance(result, dict):
        llm_answer = str(result.get("answer", ""))
        raw_output = str(result.get("raw_text", "") or llm_answer)
    else:
        llm_answer = str(result or "")
        raw_output = llm_answer
    payload = _extract_json_object(_postprocess_llm_json_output(llm_answer or raw_output))
    payload["_raw_llm_output"] = raw_output
    payload["_transcript_truncated_for_review"] = truncated
    return payload


def _apply_review(
    record: CandidateScoutRecord,
    review: dict[str, Any],
    *,
    transcript_truncated_for_review: bool,
    acceptance_threshold: float,
) -> CandidateScoutRecord:
    try:
        score = float(review.get("relevance_score"))
    except (TypeError, ValueError):
        score = 0.0
    score = round(max(0.0, min(1.0, score)), 4)
    source_duplicate = record.source_duplicate_score or 0.0
    accepted = score >= acceptance_threshold and source_duplicate < 1.0 and bool(record.transcript_path)
    decision = str(review.get("decision") or ("accept" if accepted else "reject")).strip().lower()
    if not accepted:
        decision = "reject"
    evidence = review.get("supporting_evidence", [])
    risks = review.get("risks", [])
    return record.model_copy(
        update={
            "status": "reviewed",
            "transcript_truncated_for_review": transcript_truncated_for_review,
            "llm_relevance_score": score,
            "accepted_for_queue": accepted,
            "decision": decision,
            "reason": str(review.get("reason") or "").strip() or None,
            "supporting_evidence": [str(item).strip() for item in evidence if str(item).strip()] if isinstance(evidence, list) else [],
            "risks": [str(item).strip() for item in risks if str(item).strip()] if isinstance(risks, list) else [],
            "raw_llm_output": str(review.get("_raw_llm_output") or "") or None,
        }
    )


async def run_full_candidate_scout(
    manifest: CandidateDownloadManifest,
    *,
    download_manifest_path: str | None = None,
    inventory: InventorySnapshot | None = None,
    top_n: int = DEFAULT_SCOUT_TOP_N,
    audio_extractor: AudioExtractor | None = None,
    transcript_provider: TranscriptProvider | None = None,
    embedding_func: EmbeddingFunc | None = None,
    llm_reviewer: LLMReviewer | None = None,
    acceptance_threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD,
    segment_length: int = DEFAULT_SCOUT_SEGMENT_LENGTH,
    audio_output_format: str = DEFAULT_SCOUT_AUDIO_FORMAT,
    dry_run: bool = False,
) -> CandidateScoutSnapshot:
    snapshot = run_candidate_audio_extraction(
        manifest,
        download_manifest_path=download_manifest_path,
        top_n=top_n,
        audio_extractor=audio_extractor,
        segment_length=segment_length,
        audio_output_format=audio_output_format,
        dry_run=dry_run,
    )
    if dry_run:
        return snapshot

    provider = transcript_provider or transcribe_audio_segments_with_existing_asr
    reviewed_records: list[CandidateScoutRecord] = []
    warnings = list(snapshot.warnings)
    for record in snapshot.records:
        if record.status != "audio_extracted":
            reviewed_records.append(record)
            continue
        try:
            transcript = str(await _maybe_await(provider(record)) or "").strip()
            transcript_path = _write_transcript(record, transcript)
            transcript_relevance = await compute_transcript_relevance_score(
                record,
                snapshot,
                transcript,
                embedding_func=embedding_func,
            )
            source_duplicate, source_duplicate_reason = compute_source_duplicate_score(record, manifest, inventory)
            content_redundancy, content_redundancy_reason = await compute_content_redundancy_score(
                snapshot,
                transcript,
                inventory,
                embedding_func=embedding_func,
            )
            extractability = compute_extractability_score(record, transcript)
            updated = record.model_copy(
                update={
                    "transcript_path": transcript_path,
                    "transcript_char_count": len(transcript),
                    "transcript_relevance_score": transcript_relevance,
                    "source_duplicate_score": source_duplicate,
                    "source_duplicate_reason": source_duplicate_reason,
                    "content_redundancy_score": content_redundancy,
                    "content_redundancy_reason": content_redundancy_reason,
                    "extractability_score": extractability,
                }
            )
            if not transcript:
                warnings.append(
                    CandidateScoutWarning(
                        code="empty_transcript",
                        message="ASR produced an empty transcript",
                        candidate_id=record.candidate_id,
                    )
                )
                reviewed_records.append(
                    updated.model_copy(
                        update={
                            "status": "failed",
                            "failure_reason": "empty transcript",
                            "accepted_for_queue": False,
                            "decision": "reject",
                        }
                    )
                )
                continue
            if llm_reviewer is None:
                review = await review_candidate_with_llm(updated, snapshot, transcript)
            else:
                review = await _maybe_await(llm_reviewer(updated, transcript))
            reviewed_records.append(
                _apply_review(
                    updated,
                    review,
                    transcript_truncated_for_review=bool(review.get("_transcript_truncated_for_review", False)),
                    acceptance_threshold=acceptance_threshold,
                )
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            warnings.append(
                CandidateScoutWarning(
                    code="candidate_review_failed",
                    message=message,
                    candidate_id=record.candidate_id,
                )
            )
            reviewed_records.append(
                record.model_copy(
                    update={
                        "status": "failed",
                        "failure_reason": message,
                        "accepted_for_queue": False,
                        "decision": "reject",
                    }
                )
            )

    return snapshot.model_copy(
        update={
            "records": reviewed_records,
            "warnings": warnings,
            "audio_extracted_count": sum(1 for record in reviewed_records if record.status in {"audio_extracted", "reviewed"}),
            "failed_count": sum(1 for record in reviewed_records if record.status == "failed"),
            "skipped_count": sum(1 for record in reviewed_records if record.status == "skipped"),
        }
    )


def build_selection_snapshot(
    scout_snapshot: CandidateScoutSnapshot,
    *,
    scout_snapshot_path: str | None = None,
    acceptance_threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD,
) -> CandidateSelectionSnapshot:
    records: list[CandidateSelectionRecord] = []
    for record in scout_snapshot.records:
        accepted = bool(record.accepted_for_queue)
        records.append(
            CandidateSelectionRecord(
                candidate_id=record.candidate_id,
                rank=record.rank,
                accepted_for_queue=accepted,
                decision=record.decision or ("accept" if accepted else "reject"),
                reason=record.reason or record.failure_reason,
                source_url=record.source_url,
                youtube_video_id=record.youtube_video_id,
                title=record.title,
                local_video_path=record.local_video_path,
                transcript_path=record.transcript_path,
                llm_relevance_score=record.llm_relevance_score,
                transcript_relevance_score=record.transcript_relevance_score,
                source_duplicate_score=record.source_duplicate_score,
                content_redundancy_score=record.content_redundancy_score,
                extractability_score=record.extractability_score,
                supporting_evidence=record.supporting_evidence,
                risks=record.risks,
            )
        )
    return CandidateSelectionSnapshot(
        generated_at=datetime.now(timezone.utc),
        loop_profile_id=scout_snapshot.loop_profile_id,
        scout_snapshot_path=scout_snapshot_path,
        scout_snapshot_generated_at=scout_snapshot.generated_at,
        target_gap=scout_snapshot.target_gap,
        acceptance_threshold=acceptance_threshold,
        candidate_count=len(records),
        accepted_count=sum(1 for record in records if record.accepted_for_queue),
        rejected_count=sum(1 for record in records if not record.accepted_for_queue),
        records=records,
    )


def persist_candidate_scout_snapshot(
    snapshot: CandidateScoutSnapshot,
    state_root: str | Path | None = None,
    output_path: str | Path | None = None,
    pretty: bool = True,
) -> list[Path]:
    resolved_state_root = Path(state_root or config.DEFAULT_STATE_ROOT).resolve()
    scout_dir = resolved_state_root / config.SCOUT_RUNS_DIR_NAME
    generated = snapshot.generated_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paths = [
        scout_dir / f"candidate_scout_{generated}.json",
        scout_dir / "latest_candidate_scout.json",
    ]
    if output_path:
        paths.append(Path(output_path).resolve())
    payload = snapshot.model_dump(mode="json")
    for path in paths:
        _write_json(path, payload, pretty=pretty)
    return paths


def persist_candidate_selection_snapshot(
    snapshot: CandidateSelectionSnapshot,
    state_root: str | Path | None = None,
    output_path: str | Path | None = None,
    pretty: bool = True,
) -> list[Path]:
    resolved_state_root = Path(state_root or config.DEFAULT_STATE_ROOT).resolve()
    selection_dir = resolved_state_root / config.SELECTIONS_DIR_NAME
    generated = snapshot.generated_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paths = [
        selection_dir / f"candidate_selection_{generated}.json",
        selection_dir / "latest_candidate_selection.json",
    ]
    if output_path:
        paths.append(Path(output_path).resolve())
    payload = snapshot.model_dump(mode="json")
    for path in paths:
        _write_json(path, payload, pretty=pretty)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 4 candidate scout and selection.")
    parser.add_argument(
        "--download-manifest",
        default=str(config.DEFAULT_STATE_ROOT / config.CANDIDATE_DOWNLOADS_DIR_NAME / "latest_download_manifest.json"),
    )
    parser.add_argument(
        "--inventory",
        default=str(config.DEFAULT_STATE_ROOT / config.INVENTORY_DIR_NAME / "latest_inventory.json"),
        help="Inventory snapshot used for source duplicate and content redundancy checks",
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_SCOUT_TOP_N)
    parser.add_argument("--acceptance-threshold", type=float, default=DEFAULT_ACCEPTANCE_THRESHOLD)
    parser.add_argument("--segment-length", type=int, default=DEFAULT_SCOUT_SEGMENT_LENGTH)
    parser.add_argument("--audio-format", default=DEFAULT_SCOUT_AUDIO_FORMAT)
    parser.add_argument("--state-root", default=str(config.DEFAULT_STATE_ROOT))
    parser.add_argument("--output", default=None, help="Optional extra scout output JSON path")
    parser.add_argument("--selection-output", default=None, help="Optional extra selection output JSON path")
    parser.add_argument("--audio-only", action="store_true", help="Stop after audio extraction")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compact", action="store_true")
    return parser


async def _main_async(args: argparse.Namespace) -> int:
    state_root = args.state_root
    manifest_path = Path(args.download_manifest).resolve()
    manifest = load_download_manifest(manifest_path)
    inventory = None if args.audio_only or args.dry_run else load_inventory_snapshot(args.inventory)
    if args.audio_only:
        snapshot = run_candidate_audio_extraction(
            manifest,
            download_manifest_path=str(manifest_path),
            top_n=max(1, args.top_n),
            segment_length=max(1, args.segment_length),
            audio_output_format=args.audio_format,
            dry_run=args.dry_run,
        )
    else:
        snapshot = await run_full_candidate_scout(
            manifest,
            download_manifest_path=str(manifest_path),
            inventory=inventory,
            top_n=max(1, args.top_n),
            acceptance_threshold=max(0.0, min(1.0, args.acceptance_threshold)),
            segment_length=max(1, args.segment_length),
            audio_output_format=args.audio_format,
            dry_run=args.dry_run,
        )
    scout_paths = persist_candidate_scout_snapshot(
        snapshot,
        state_root=state_root,
        output_path=args.output,
        pretty=not args.compact,
    )
    selection = None
    selection_paths: list[Path] = []
    if not args.audio_only:
        selection = build_selection_snapshot(
            snapshot,
            scout_snapshot_path=str(scout_paths[0].resolve()),
            acceptance_threshold=max(0.0, min(1.0, args.acceptance_threshold)),
        )
        selection_paths = persist_candidate_selection_snapshot(
            selection,
            state_root=state_root,
            output_path=args.selection_output,
            pretty=not args.compact,
        )

    print(f"Scout profile: {snapshot.loop_profile_id}")
    print(f"Target gap: {snapshot.target_gap.gap_id} {snapshot.target_gap.topic_key} {snapshot.target_gap.facet}")
    print(f"Selected candidates: {snapshot.selected_candidate_count}")
    print(f"Audio extracted: {snapshot.audio_extracted_count}")
    print(f"Failed: {snapshot.failed_count}")
    print(f"Skipped: {snapshot.skipped_count}")
    if selection is not None:
        print(f"Accepted for queue: {selection.accepted_count}")
        print(f"Rejected: {selection.rejected_count}")
    print(f"Warnings: {len(snapshot.warnings)}")
    for record in snapshot.records:
        print(f"{record.rank}. {record.status.upper()} {record.candidate_id} {record.title}")
        print(f"   audio segments: {record.audio_segment_count}")
        if record.llm_relevance_score is not None:
            print(f"   LLM relevance: {record.llm_relevance_score:.4f}")
        if record.accepted_for_queue is not None:
            print(f"   accepted: {record.accepted_for_queue}")
        if record.audio_working_dir:
            print(f"   audio dir: {record.audio_working_dir}")
        if record.transcript_path:
            print(f"   transcript: {record.transcript_path}")
        if record.failure_reason:
            print(f"   reason: {record.failure_reason}")
    for warning in snapshot.warnings:
        print(f"WARNING {warning.code}: {warning.message}")
    for path in scout_paths + selection_paths:
        print(f"Wrote: {path}")
    return 1 if snapshot.failed_count else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
