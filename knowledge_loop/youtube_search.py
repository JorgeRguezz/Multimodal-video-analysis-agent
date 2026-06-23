from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_build._llm import local_llm_config

from . import config
from .models import (
    SearchIntent,
    SearchPlanSnapshot,
    YouTubeCandidate,
    YouTubeQuotaEstimate,
    YouTubeSearchRunSnapshot,
    YouTubeSearchWarning,
)
from .prompts import SYSTEM_YOUTUBE_CANDIDATE_RANKER, USER_YOUTUBE_CANDIDATE_RANK_TEMPLATE
from .search_plan import _extract_json_object


YOUTUBE_SEARCH_ENDPOINT = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_ENDPOINT = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_WATCH_URL = "https://www.youtube.com/watch?v={video_id}"
SEARCH_LIST_QUOTA_COST = 100
VIDEOS_LIST_QUOTA_COST = 1
YOUTUBE_BATCH_SIZE = 50
DEFAULT_LLM_CANDIDATE_LIMIT = 10

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "best",
    "for",
    "from",
    "guide",
    "how",
    "in",
    "is",
    "league",
    "legends",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: dict[str, Any], pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2 if pretty else None, ensure_ascii=False)


def load_search_plan_snapshot(path: str | Path) -> SearchPlanSnapshot:
    return SearchPlanSnapshot.model_validate(_read_json(Path(path)))


def _request_json(url: str, params: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    query = urllib.parse.urlencode({key: value for key, value in params.items() if value is not None})
    request = urllib.request.Request(f"{url}?{query}", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"YouTube API HTTP {exc.code}: {body}") from exc


def _parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_youtube_duration(value: str | None) -> int | None:
    if not value:
        return None
    match = re.fullmatch(
        r"P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?",
        value,
    )
    if not match:
        return None
    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _candidate_id(video_id: str) -> str:
    return "cand-" + hashlib.md5(f"youtube:{video_id}".encode("utf-8")).hexdigest()[:16]


def _tokenize(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", str(text or "").lower()))
    return {token for token in tokens if token not in STOPWORDS and len(token) > 1}


def _overlap_score(needles: set[str], text: str) -> float:
    if not needles:
        return 0.0
    haystack = _tokenize(text)
    if not haystack:
        return 0.0
    return len(needles & haystack) / max(1, len(needles))


def _normalize_by_max(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return max(0.0, min(1.0, value / max_value))


def _freshness_score(publish_date: datetime | None, now: datetime) -> float:
    if publish_date is None:
        return 0.35
    if publish_date.tzinfo is None:
        publish_date = publish_date.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - publish_date.astimezone(timezone.utc)).total_seconds() / 86400)
    if age_days <= 30:
        return 1.0
    if age_days <= 180:
        return 0.85
    if age_days <= 365:
        return 0.70
    if age_days <= 730:
        return 0.45
    if age_days <= 1095:
        return 0.25
    return 0.10


def _duration_fit_score(duration_seconds: int | None) -> float:
    if duration_seconds is None:
        return 0.40
    if duration_seconds <= 0:
        return 0.0
    if duration_seconds < 180:
        return max(0.15, duration_seconds / 180)
    if duration_seconds <= 1200:
        return 1.0
    if duration_seconds <= 3600:
        return max(0.20, 1.0 - ((duration_seconds - 1200) / 2400) * 0.80)
    return 0.20


def _channel_quality_hint(candidate: YouTubeCandidate) -> float:
    text = " ".join(
        [
            candidate.title,
            candidate.description,
            candidate.channel_title or "",
        ]
    ).lower()
    hints = (
        "guide",
        "coach",
        "coaching",
        "challenger",
        "explained",
        "tutorial",
        "tips",
        "tricks",
        "season",
        "patch",
        "educational",
    )
    hits = sum(1 for hint in hints if hint in text)
    return min(1.0, hits / 3)


def _search_youtube_api(
    *,
    api_key: str,
    query: str,
    max_results: int,
    order: str,
    published_after: str | None,
    published_before: str | None,
    region_code: str | None,
    relevance_language: str | None,
    video_duration: str | None,
) -> dict[str, Any]:
    return _request_json(
        YOUTUBE_SEARCH_ENDPOINT,
        {
            "key": api_key,
            "part": "snippet",
            "type": "video",
            "q": query,
            "maxResults": max(1, min(50, max_results)),
            "order": order,
            "publishedAfter": published_after,
            "publishedBefore": published_before,
            "regionCode": region_code,
            "relevanceLanguage": relevance_language,
            "videoDuration": video_duration,
        },
    )


def _fetch_video_metadata_api(api_key: str, video_ids: list[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for idx in range(0, len(video_ids), YOUTUBE_BATCH_SIZE):
        batch = video_ids[idx : idx + YOUTUBE_BATCH_SIZE]
        if not batch:
            continue
        payload = _request_json(
            YOUTUBE_VIDEOS_ENDPOINT,
            {
                "key": api_key,
                "part": "snippet,contentDetails,statistics",
                "id": ",".join(batch),
            },
        )
        raw_items = payload.get("items", [])
        if isinstance(raw_items, list):
            items.extend([item for item in raw_items if isinstance(item, dict)])
    return items


def collect_video_ids_from_search_results(
    search_results_by_intent: list[tuple[SearchIntent, dict[str, Any]]],
) -> dict[str, set[str]]:
    matched: dict[str, set[str]] = {}
    for intent, payload in search_results_by_intent:
        items = payload.get("items", [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id", {})
            if not isinstance(item_id, dict):
                continue
            video_id = item_id.get("videoId")
            if not video_id:
                continue
            matched.setdefault(str(video_id), set()).add(intent.query)
    return matched


def candidates_from_video_items(
    video_items: list[dict[str, Any]],
    matched_intents_by_video_id: dict[str, set[str]],
) -> list[YouTubeCandidate]:
    candidates: list[YouTubeCandidate] = []
    for item in video_items:
        video_id = str(item.get("id") or "").strip()
        if not video_id:
            continue
        snippet = item.get("snippet", {}) if isinstance(item.get("snippet"), dict) else {}
        content = item.get("contentDetails", {}) if isinstance(item.get("contentDetails"), dict) else {}
        stats = item.get("statistics", {}) if isinstance(item.get("statistics"), dict) else {}
        candidate = YouTubeCandidate(
            candidate_id=_candidate_id(video_id),
            youtube_video_id=video_id,
            source_url=YOUTUBE_WATCH_URL.format(video_id=video_id),
            title=str(snippet.get("title") or "").strip(),
            description=str(snippet.get("description") or "").strip(),
            channel_id=str(snippet.get("channelId") or "").strip() or None,
            channel_title=str(snippet.get("channelTitle") or "").strip() or None,
            publish_date=_parse_datetime(snippet.get("publishedAt")),
            duration_seconds=parse_youtube_duration(content.get("duration")),
            view_count=_parse_int(stats.get("viewCount")),
            like_count=_parse_int(stats.get("likeCount")),
            matched_intents=sorted(matched_intents_by_video_id.get(video_id, set())),
            raw_metadata=item,
        )
        candidates.append(candidate)
    return candidates


def _ranking_tokens(search_plan: SearchPlanSnapshot) -> set[str]:
    text = " ".join(
        [
            search_plan.target_gap.topic_key.replace("::", " "),
            search_plan.target_gap.facet,
            search_plan.target_gap.evidence_summary,
            " ".join(intent.query for intent in search_plan.search_intents),
        ]
    )
    return _tokenize(text)


def score_candidates_with_metadata(
    candidates: list[YouTubeCandidate],
    search_plan: SearchPlanSnapshot,
    *,
    now: datetime | None = None,
) -> list[YouTubeCandidate]:
    now = now or datetime.now(timezone.utc)
    topic_tokens = _ranking_tokens(search_plan)
    max_views_log = max((math.log1p(candidate.view_count or 0) for candidate in candidates), default=0.0)
    max_likes_log = max((math.log1p(candidate.like_count or 0) for candidate in candidates), default=0.0)

    view_velocities: list[float] = []
    like_velocities: list[float] = []
    for candidate in candidates:
        age_days = 365.0
        if candidate.publish_date is not None:
            publish_date = candidate.publish_date
            if publish_date.tzinfo is None:
                publish_date = publish_date.replace(tzinfo=timezone.utc)
            age_days = max(1.0, (now - publish_date.astimezone(timezone.utc)).total_seconds() / 86400)
        view_velocities.append((candidate.view_count or 0) / age_days)
        like_velocities.append((candidate.like_count or 0) / age_days)
    max_view_velocity_log = max((math.log1p(value) for value in view_velocities), default=0.0)
    max_like_velocity_log = max((math.log1p(value) for value in like_velocities), default=0.0)

    scored: list[YouTubeCandidate] = []
    total_intents = max(1, len(search_plan.search_intents))
    for idx, candidate in enumerate(candidates):
        title_relevance = _overlap_score(topic_tokens, candidate.title)
        description_relevance = _overlap_score(topic_tokens, candidate.description)
        intent_coverage = min(1.0, len(candidate.matched_intents) / total_intents)
        freshness = _freshness_score(candidate.publish_date, now)
        duration_fit = _duration_fit_score(candidate.duration_seconds)

        absolute_views = _normalize_by_max(math.log1p(candidate.view_count or 0), max_views_log)
        absolute_likes = _normalize_by_max(math.log1p(candidate.like_count or 0), max_likes_log)
        velocity_views = _normalize_by_max(math.log1p(view_velocities[idx]), max_view_velocity_log)
        velocity_likes = _normalize_by_max(math.log1p(like_velocities[idx]), max_like_velocity_log)
        absolute_engagement = 0.75 * absolute_views + 0.25 * absolute_likes
        velocity_engagement = 0.75 * velocity_views + 0.25 * velocity_likes
        engagement = 0.55 * absolute_engagement + 0.45 * velocity_engagement

        channel_hint = _channel_quality_hint(candidate)
        metadata_score = (
            0.30 * title_relevance
            + 0.18 * description_relevance
            + 0.12 * intent_coverage
            + 0.15 * freshness
            + 0.15 * engagement
            + 0.08 * duration_fit
            + 0.02 * channel_hint
        )
        reasons = []
        if title_relevance >= 0.30:
            reasons.append("title_matches_gap")
        if description_relevance >= 0.20:
            reasons.append("description_matches_gap")
        if freshness >= 0.70:
            reasons.append("fresh_or_recent")
        if duration_fit >= 0.90:
            reasons.append("duration_3_to_20_minutes")
        if engagement >= 0.60:
            reasons.append("strong_engagement")
        if intent_coverage > 1 / total_intents:
            reasons.append("matched_multiple_intents")

        candidate.metadata_components = {
            "title_relevance": round(title_relevance, 4),
            "description_relevance": round(description_relevance, 4),
            "intent_coverage": round(intent_coverage, 4),
            "freshness_score": round(freshness, 4),
            "engagement_score": round(engagement, 4),
            "duration_fit_score": round(duration_fit, 4),
            "channel_quality_hint": round(channel_hint, 4),
        }
        candidate.metadata_score = round(max(0.0, min(1.0, metadata_score)), 4)
        candidate.final_score = candidate.metadata_score
        candidate.ranking_reasons = reasons
        scored.append(candidate)

    return sorted(scored, key=lambda candidate: (-candidate.metadata_score, candidate.title))


def _candidate_for_prompt(candidate: YouTubeCandidate) -> dict[str, Any]:
    description = re.sub(r"\s+", " ", candidate.description).strip()
    if len(description) > 650:
        description = description[:640].rstrip() + "..."
    return {
        "candidate_id": candidate.candidate_id,
        "title": candidate.title,
        "description": description,
        "channel_title": candidate.channel_title,
        "publish_date": candidate.publish_date.isoformat() if candidate.publish_date else None,
        "duration_seconds": candidate.duration_seconds,
        "view_count": candidate.view_count,
        "like_count": candidate.like_count,
        "matched_intents": candidate.matched_intents,
        "metadata_score": candidate.metadata_score,
        "metadata_components": candidate.metadata_components,
    }


def _format_llm_ranking_prompt(
    search_plan: SearchPlanSnapshot,
    candidates: list[YouTubeCandidate],
) -> str:
    intents = "\n".join(f"- {intent.query}: {intent.purpose}" for intent in search_plan.search_intents)
    compact_candidates = [_candidate_for_prompt(candidate) for candidate in candidates]
    return USER_YOUTUBE_CANDIDATE_RANK_TEMPLATE.format(
        gap_id=search_plan.target_gap.gap_id,
        gap_kind=search_plan.target_gap.gap_kind,
        topic_key=search_plan.target_gap.topic_key,
        facet=search_plan.target_gap.facet,
        priority=search_plan.target_gap.priority,
        evidence_summary=search_plan.target_gap.evidence_summary,
        reasons=", ".join(search_plan.target_gap.reasons) if search_plan.target_gap.reasons else "none",
        search_intents=intents,
        candidates=json.dumps(compact_candidates, indent=2, ensure_ascii=False),
    )


def _strip_markdown_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else stripped


def _postprocess_llm_ranking_output(llm_text: str) -> str:
    """Prefer final-channel JSON when local model output includes analysis traces."""
    text = str(llm_text or "").strip()
    if not text:
        return text

    final_markers = (
        "<|channel|>final<|message|>",
        "<|start|>assistant<|channel|>final<|message|>",
        "assistant<|channel|>final<|message|>",
    )
    final_start = -1
    final_marker = ""
    for marker in final_markers:
        idx = text.rfind(marker)
        if idx > final_start:
            final_start = idx
            final_marker = marker
    if final_start >= 0:
        text = text[final_start + len(final_marker) :]

    for end_marker in ("<|return|>", "<|end|>", "<|start|>"):
        end_idx = text.find(end_marker)
        if end_idx >= 0:
            text = text[:end_idx]

    text = _strip_markdown_json_fence(text)
    if "{" in text:
        return text[text.find("{") :].strip()
    return text.strip()


def _parse_llm_candidate_scores(
    llm_text: str,
) -> tuple[dict[str, tuple[float, str]], list[YouTubeSearchWarning]]:
    cleaned_text = _postprocess_llm_ranking_output(llm_text)
    try:
        payload = _extract_json_object(cleaned_text)
    except Exception as exc:  # noqa: BLE001
        return {}, [
            YouTubeSearchWarning(
                code="llm_ranking_json_parse_failed",
                message=f"Could not parse LLM ranking JSON: {exc}",
            )
        ]
    raw_ranked = payload.get("ranked_candidates", [])
    if not isinstance(raw_ranked, list):
        return {}, [
            YouTubeSearchWarning(
                code="invalid_ranked_candidates",
                message="LLM output field 'ranked_candidates' is not a list",
            )
        ]
    scores: dict[str, tuple[float, str]] = {}
    warnings: list[YouTubeSearchWarning] = []
    for idx, item in enumerate(raw_ranked):
        if not isinstance(item, dict):
            warnings.append(
                YouTubeSearchWarning(
                    code="invalid_llm_ranking_entry",
                    message=f"Ranking entry at index {idx} is not an object",
                )
            )
            continue
        candidate_id = str(item.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        try:
            score = float(item.get("llm_score"))
        except (TypeError, ValueError):
            warnings.append(
                YouTubeSearchWarning(
                    code="invalid_llm_score",
                    message=f"Invalid llm_score for candidate {candidate_id}",
                    candidate_id=candidate_id,
                )
            )
            continue
        score = max(0.0, min(1.0, score))
        reason = str(item.get("reason") or "").strip()
        scores[candidate_id] = (score, reason)
    return scores, warnings


async def rerank_candidates_with_llm(
    search_plan: SearchPlanSnapshot,
    candidates: list[YouTubeCandidate],
) -> tuple[list[YouTubeCandidate], list[YouTubeSearchWarning], str | None]:
    if not candidates:
        return [], [YouTubeSearchWarning(code="no_candidates_for_llm", message="No candidates to rank")], None

    prompt = _format_llm_ranking_prompt(search_plan, candidates)
    try:
        result = await local_llm_config.best_model_func(
            prompt,
            system_prompt=SYSTEM_YOUTUBE_CANDIDATE_RANKER,
            max_tokens=5400,
            temperature=0.1,
            top_p=1.0,
            top_k=0,
            repeat_penalty=1.08,
            return_metadata=True,
        )
    except Exception as exc:  # noqa: BLE001
        return candidates, [
            YouTubeSearchWarning(
                code="llm_ranking_call_failed",
                message=f"Local LLM ranking call failed: {exc}",
            )
        ], None

    if isinstance(result, dict):
        llm_answer = str(result.get("answer", ""))
        raw_output = str(result.get("raw_text", "") or llm_answer)
    else:
        llm_answer = str(result or "")
        raw_output = llm_answer

    scores, warnings = _parse_llm_candidate_scores(llm_answer or raw_output)
    ranked: list[YouTubeCandidate] = []
    for candidate in candidates:
        if candidate.candidate_id not in scores:
            warnings.append(
                YouTubeSearchWarning(
                    code="candidate_missing_from_llm_ranking",
                    message="Candidate was not scored by the LLM and will receive a low final score",
                    candidate_id=candidate.candidate_id,
                )
            )
            candidate.llm_score = None
            candidate.final_score = round(0.45 * candidate.metadata_score, 4)
        else:
            llm_score, reason = scores[candidate.candidate_id]
            candidate.llm_score = round(llm_score, 4)
            candidate.llm_reason = reason or None
            candidate.final_score = round(0.45 * candidate.metadata_score + 0.55 * llm_score, 4)
        ranked.append(candidate)

    return sorted(ranked, key=lambda candidate: (-candidate.final_score, candidate.title)), warnings, raw_output


def estimate_quota(search_calls: int, unique_video_ids: int) -> YouTubeQuotaEstimate:
    video_list_calls = math.ceil(unique_video_ids / YOUTUBE_BATCH_SIZE) if unique_video_ids else 0
    return YouTubeQuotaEstimate(
        search_calls=search_calls,
        video_list_calls=video_list_calls,
        estimated_units=search_calls * SEARCH_LIST_QUOTA_COST + video_list_calls * VIDEOS_LIST_QUOTA_COST,
    )


async def run_youtube_search(
    search_plan: SearchPlanSnapshot,
    *,
    api_key: str,
    search_plan_path: str | None = None,
    max_results_per_intent: int = 10,
    top_n: int = 10,
    llm_candidate_limit: int = DEFAULT_LLM_CANDIDATE_LIMIT,
    order: str = "relevance",
    published_after: str | None = None,
    published_before: str | None = None,
    region_code: str | None = None,
    relevance_language: str | None = None,
    video_duration: str | None = None,
    keep_raw_output: bool = False,
) -> YouTubeSearchRunSnapshot:
    warnings: list[YouTubeSearchWarning] = []
    search_results: list[tuple[SearchIntent, dict[str, Any]]] = []
    for intent in search_plan.search_intents:
        try:
            payload = _search_youtube_api(
                api_key=api_key,
                query=intent.query,
                max_results=max_results_per_intent,
                order=order,
                published_after=published_after,
                published_before=published_before,
                region_code=region_code,
                relevance_language=relevance_language,
                video_duration=video_duration,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                YouTubeSearchWarning(
                    code="youtube_search_failed",
                    message=f"Search failed for query {intent.query!r}: {exc}",
                )
            )
            continue
        search_results.append((intent, payload))

    matched_intents = collect_video_ids_from_search_results(search_results)
    video_ids = sorted(matched_intents)
    quota = estimate_quota(search_calls=len(search_plan.search_intents), unique_video_ids=len(video_ids))
    if not video_ids:
        return YouTubeSearchRunSnapshot(
            generated_at=datetime.now(timezone.utc),
            loop_profile_id=search_plan.loop_profile_id,
            search_plan_path=search_plan_path,
            target_gap=search_plan.target_gap,
            search_intents=search_plan.search_intents,
            raw_candidate_count=0,
            llm_ranked_candidate_count=0,
            candidate_count=0,
            ranked_candidates=[],
            warnings=warnings
            + [YouTubeSearchWarning(code="no_video_ids", message="No video IDs returned by YouTube search")],
            quota_estimate=quota,
        )

    try:
        video_items = _fetch_video_metadata_api(api_key, video_ids)
    except Exception as exc:  # noqa: BLE001
        return YouTubeSearchRunSnapshot(
            generated_at=datetime.now(timezone.utc),
            loop_profile_id=search_plan.loop_profile_id,
            search_plan_path=search_plan_path,
            target_gap=search_plan.target_gap,
            search_intents=search_plan.search_intents,
            raw_candidate_count=len(video_ids),
            llm_ranked_candidate_count=0,
            candidate_count=0,
            ranked_candidates=[],
            warnings=warnings
            + [YouTubeSearchWarning(code="youtube_videos_list_failed", message=f"Metadata fetch failed: {exc}")],
            quota_estimate=quota,
        )

    candidates = candidates_from_video_items(video_items, matched_intents)
    scored = score_candidates_with_metadata(candidates, search_plan)
    llm_pool = scored[: max(1, llm_candidate_limit)]
    if len(scored) > len(llm_pool):
        warnings.append(
            YouTubeSearchWarning(
                code="metadata_prefilter_applied",
                message=(
                    f"Metadata prefilter kept {len(llm_pool)} of {len(scored)} candidates "
                    "for mandatory LLM ranking"
                ),
            )
        )

    llm_ranked, llm_warnings, raw_llm_output = await rerank_candidates_with_llm(search_plan, llm_pool)
    warnings.extend(llm_warnings)
    ranked = llm_ranked[: max(1, top_n)]

    return YouTubeSearchRunSnapshot(
        generated_at=datetime.now(timezone.utc),
        loop_profile_id=search_plan.loop_profile_id,
        search_plan_path=search_plan_path,
        target_gap=search_plan.target_gap,
        search_intents=search_plan.search_intents,
        raw_candidate_count=len(candidates),
        llm_ranked_candidate_count=len(llm_pool),
        candidate_count=len(ranked),
        ranked_candidates=ranked,
        warnings=warnings,
        quota_estimate=quota,
        raw_llm_output=raw_llm_output if keep_raw_output else None,
    )


def persist_youtube_search_snapshot(
    snapshot: YouTubeSearchRunSnapshot,
    state_root: str | Path | None = None,
    output_path: str | Path | None = None,
    pretty: bool = True,
) -> list[Path]:
    resolved_state_root = Path(state_root or config.DEFAULT_STATE_ROOT).resolve()
    search_dir = resolved_state_root / config.SEARCH_RUNS_DIR_NAME
    generated = snapshot.generated_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paths = [
        search_dir / f"youtube_search_{generated}.json",
        search_dir / "latest_youtube_search.json",
    ]
    if output_path:
        paths.append(Path(output_path).resolve())
    payload = snapshot.model_dump(mode="json")
    for path in paths:
        _write_json(path, payload, pretty=pretty)
    return paths


def build_dry_run_snapshot(
    search_plan: SearchPlanSnapshot,
    *,
    search_plan_path: str | None,
    max_results_per_intent: int,
) -> YouTubeSearchRunSnapshot:
    estimated_unique = len(search_plan.search_intents) * max_results_per_intent
    return YouTubeSearchRunSnapshot(
        generated_at=datetime.now(timezone.utc),
        loop_profile_id=search_plan.loop_profile_id,
        search_plan_path=search_plan_path,
        target_gap=search_plan.target_gap,
        search_intents=search_plan.search_intents,
        raw_candidate_count=0,
        llm_ranked_candidate_count=0,
        candidate_count=0,
        ranked_candidates=[],
        warnings=[
            YouTubeSearchWarning(
                code="dry_run",
                message="Dry run only; no YouTube API or LLM ranking calls were made",
            )
        ],
        quota_estimate=estimate_quota(len(search_plan.search_intents), estimated_unique),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search YouTube and rank video candidates for a search plan.")
    parser.add_argument(
        "--search-plan",
        default=str(config.DEFAULT_STATE_ROOT / config.SEARCH_PLANS_DIR_NAME / "latest_search_plan.json"),
        help="Search plan JSON path",
    )
    parser.add_argument("--api-key", default=None, help="YouTube API key; defaults to YOUTUBE_API_KEY")
    parser.add_argument("--max-results-per-intent", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--llm-candidate-limit", type=int, default=DEFAULT_LLM_CANDIDATE_LIMIT)
    parser.add_argument("--order", default="relevance")
    parser.add_argument("--published-after", default=None)
    parser.add_argument("--published-before", default=None)
    parser.add_argument("--region-code", default=None)
    parser.add_argument("--relevance-language", default=None)
    parser.add_argument("--video-duration", default=None)
    parser.add_argument("--state-root", default=str(config.DEFAULT_STATE_ROOT))
    parser.add_argument("--output", default=None, help="Optional extra output JSON path")
    parser.add_argument("--keep-raw-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compact", action="store_true")
    return parser


async def _main_async(args: argparse.Namespace) -> int:
    search_plan_path = Path(args.search_plan)
    search_plan = load_search_plan_snapshot(search_plan_path)
    if args.dry_run:
        snapshot = build_dry_run_snapshot(
            search_plan,
            search_plan_path=str(search_plan_path.resolve()),
            max_results_per_intent=max(1, min(50, args.max_results_per_intent)),
        )
    else:
        api_key = args.api_key or os.environ.get("YOUTUBE_API_KEY")
        if not api_key:
            raise SystemExit("Missing YouTube API key. Set YOUTUBE_API_KEY or pass --api-key.")
        snapshot = await run_youtube_search(
            search_plan,
            api_key=api_key,
            search_plan_path=str(search_plan_path.resolve()),
            max_results_per_intent=max(1, min(50, args.max_results_per_intent)),
            top_n=max(1, args.top_n),
            llm_candidate_limit=max(1, args.llm_candidate_limit),
            order=args.order,
            published_after=args.published_after,
            published_before=args.published_before,
            region_code=args.region_code,
            relevance_language=args.relevance_language,
            video_duration=args.video_duration,
            keep_raw_output=args.keep_raw_output,
        )

    paths = persist_youtube_search_snapshot(
        snapshot,
        state_root=args.state_root,
        output_path=args.output,
        pretty=not args.compact,
    )
    print(f"YouTube search profile: {snapshot.loop_profile_id}")
    print(f"Target gap: {snapshot.target_gap.gap_id} {snapshot.target_gap.topic_key} {snapshot.target_gap.facet}")
    print(f"Raw candidates: {snapshot.raw_candidate_count}")
    print(f"LLM-ranked candidates: {snapshot.llm_ranked_candidate_count}")
    print(f"Ranked candidates: {snapshot.candidate_count}")
    print(f"Estimated quota units: {snapshot.quota_estimate.estimated_units}")
    print(f"Warnings: {len(snapshot.warnings)}")
    for candidate in snapshot.ranked_candidates[:10]:
        print(f"{candidate.final_score:.4f} {candidate.title} ({candidate.source_url})")
    for warning in snapshot.warnings:
        print(f"WARNING {warning.code}: {warning.message}")
    for path in paths:
        print(f"Wrote: {path}")

    llm_failed = any(warning.code.startswith("llm_ranking") for warning in snapshot.warnings)
    return 2 if llm_failed and not args.dry_run else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
