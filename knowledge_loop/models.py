from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoopGameKnowledgeProfile(BaseModel):
    """Loop-level game knowledge organization, separate from extraction profiles."""

    model_config = ConfigDict(frozen=True)

    id: str
    display_name: str
    extraction_game_id: str
    facets: tuple[str, ...]
    default_facet: str
    facet_keywords: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    topic_prefix_by_facet: dict[str, str] = Field(default_factory=dict)
    entity_stopwords: tuple[str, ...] = ()
    generic_topic_key: str | None = None
    target_chunks_per_facet: int = 6
    target_sources_per_facet: int = 3
    core_gap_facets: tuple[str, ...] = ()
    noisy_topic_tokens: tuple[str, ...] = ()
    covered_threshold: float = 0.85
    min_actionability_score: float = 0.45
    min_actionable_chunks: int = 2
    min_actionable_videos: int = 2
    missing_facet_min_chunks: int = 3
    missing_facet_min_videos: int = 2
    stale_after_days: int = 180

    @field_validator("default_facet")
    @classmethod
    def _default_facet_must_exist(cls, value: str, info):
        facets = info.data.get("facets", ())
        if facets and value not in facets:
            raise ValueError(f"default_facet={value!r} is not in facets")
        return value

    @field_validator("core_gap_facets")
    @classmethod
    def _core_gap_facets_must_exist(cls, value: tuple[str, ...], info):
        facets = info.data.get("facets", ())
        unknown = [facet for facet in value if facet not in facets]
        if unknown:
            raise ValueError(f"core_gap_facets contains unknown facets: {unknown}")
        return value


class InventoryWarning(BaseModel):
    code: str
    message: str
    video_name: str | None = None
    path: str | None = None


class InventorySourceRef(BaseModel):
    source_kind: Literal["chunk", "graph"]
    video_name: str
    chunk_id: str | None = None
    segment_ids: list[str] = Field(default_factory=list)
    graph_node: str | None = None
    time_span: str | None = None
    text_preview: str | None = None


class VideoInventory(BaseModel):
    video_name: str
    cache_dir: str
    source_path: str | None = None
    chunk_count: int = 0
    segment_count: int = 0
    frame_count: int = 0
    graph_node_count: int = 0
    graph_edge_count: int = 0
    freshness_days: int | None = None


class TopicFacetInventory(BaseModel):
    topic_key: str
    facet: str
    video_count: int
    chunk_count: int
    graph_entity_count: int
    source_redundancy: int
    source_diversity_score: float
    weak_redundancy_score: float
    coverage_score: float
    freshness_days: int | None = None
    supporting_videos: list[str] = Field(default_factory=list)
    evidence: list[InventorySourceRef] = Field(default_factory=list)


class InventorySnapshot(BaseModel):
    schema_version: int = 1
    generated_at: datetime
    loop_profile_id: str
    extraction_game_id: str
    source_root: str
    videos: dict[str, VideoInventory] = Field(default_factory=dict)
    topic_facets: list[TopicFacetInventory] = Field(default_factory=list)
    warnings: list[InventoryWarning] = Field(default_factory=list)
    global_graph_node_count: int | None = None
    global_graph_edge_count: int | None = None


GapStatus = Literal["open", "active", "partially_covered", "covered", "blocked"]
GapKind = Literal["low_coverage", "missing_facet"]


class GapWarning(BaseModel):
    code: str
    message: str
    topic_key: str | None = None


class KnowledgeGap(BaseModel):
    gap_id: str
    gap_kind: GapKind
    topic_key: str
    facet: str
    priority: float
    status: GapStatus = "open"
    coverage_score: float
    coverage_gap_score: float
    missing_facet_score: float
    freshness_days: int | None = None
    freshness_score: float | None = None
    weak_redundancy_score: float
    actionability_score: float
    video_count: int
    chunk_count: int
    source_redundancy: int
    graph_entity_count: int = 0
    supporting_videos: list[str] = Field(default_factory=list)
    source_inventory_topic_keys: list[str] = Field(default_factory=list)
    evidence_summary: str
    reasons: list[str] = Field(default_factory=list)


class GapSnapshot(BaseModel):
    schema_version: int = 1
    generated_at: datetime
    loop_profile_id: str
    inventory_generated_at: datetime | None = None
    inventory_source_path: str | None = None
    raw_bucket_count: int = 0
    actionable_bucket_count: int = 0
    filtered_bucket_count: int = 0
    gap_count: int = 0
    include_missing_facets: bool = False
    gaps: list[KnowledgeGap] = Field(default_factory=list)
    warnings: list[GapWarning] = Field(default_factory=list)


class SearchPlanWarning(BaseModel):
    code: str
    message: str


class SearchIntent(BaseModel):
    query: str
    purpose: str
    source_type: str
    expected_signal: str | None = None

    @field_validator("query", "purpose", "source_type")
    @classmethod
    def _required_text(cls, value: str):
        value = str(value or "").strip()
        if not value:
            raise ValueError("field must be non-empty")
        return value

    @field_validator("expected_signal")
    @classmethod
    def _optional_text(cls, value: str | None):
        if value is None:
            return None
        value = str(value).strip()
        return value or None


class SearchPlanSnapshot(BaseModel):
    schema_version: int = 1
    generated_at: datetime
    loop_profile_id: str
    target_gap: KnowledgeGap
    search_intents: list[SearchIntent] = Field(default_factory=list)
    warnings: list[SearchPlanWarning] = Field(default_factory=list)
    raw_llm_output: str | None = None


class YouTubeSearchWarning(BaseModel):
    code: str
    message: str
    candidate_id: str | None = None


class YouTubeQuotaEstimate(BaseModel):
    search_calls: int = 0
    video_list_calls: int = 0
    estimated_units: int = 0


class YouTubeCandidate(BaseModel):
    candidate_id: str
    youtube_video_id: str
    source_url: str
    title: str
    description: str = ""
    channel_id: str | None = None
    channel_title: str | None = None
    publish_date: datetime | None = None
    duration_seconds: int | None = None
    view_count: int | None = None
    like_count: int | None = None
    matched_intents: list[str] = Field(default_factory=list)
    metadata_score: float = 0.0
    llm_score: float | None = None
    final_score: float = 0.0
    ranking_reasons: list[str] = Field(default_factory=list)
    llm_reason: str | None = None
    metadata_components: dict[str, float] = Field(default_factory=dict)
    raw_metadata: dict = Field(default_factory=dict)


class YouTubeSearchRunSnapshot(BaseModel):
    schema_version: int = 1
    generated_at: datetime
    loop_profile_id: str
    search_plan_path: str | None = None
    target_gap: KnowledgeGap
    search_intents: list[SearchIntent] = Field(default_factory=list)
    backend: str = "youtube_api"
    raw_candidate_count: int = 0
    llm_ranked_candidate_count: int = 0
    candidate_count: int = 0
    ranked_candidates: list[YouTubeCandidate] = Field(default_factory=list)
    warnings: list[YouTubeSearchWarning] = Field(default_factory=list)
    quota_estimate: YouTubeQuotaEstimate = Field(default_factory=YouTubeQuotaEstimate)
    raw_llm_output: str | None = None


DownloadStatus = Literal["downloaded", "failed", "skipped"]


class CandidateDownloadWarning(BaseModel):
    code: str
    message: str
    candidate_id: str | None = None


class CandidateDownloadRecord(BaseModel):
    candidate_id: str
    rank: int
    status: DownloadStatus
    source_url: str
    youtube_video_id: str | None = None
    title: str
    channel_title: str | None = None
    publish_date: datetime | None = None
    duration_seconds: int | None = None
    final_score: float = 0.0
    llm_score: float | None = None
    metadata_score: float = 0.0
    download_dir: str
    local_video_path: str | None = None
    yt_dlp_info_path: str | None = None
    cookie_source: str | None = None
    downloaded_at: datetime | None = None
    failure_reason: str | None = None


class CandidateDownloadManifest(BaseModel):
    schema_version: int = 1
    generated_at: datetime
    loop_profile_id: str
    search_run_path: str | None = None
    search_run_generated_at: datetime | None = None
    target_gap: KnowledgeGap
    requested_top_n: int
    selected_candidate_count: int = 0
    successful_download_count: int = 0
    failed_download_count: int = 0
    skipped_download_count: int = 0
    download_root: str
    records: list[CandidateDownloadRecord] = Field(default_factory=list)
    warnings: list[CandidateDownloadWarning] = Field(default_factory=list)


ScoutStatus = Literal["audio_extracted", "reviewed", "failed", "skipped"]


class CandidateScoutWarning(BaseModel):
    code: str
    message: str
    candidate_id: str | None = None


class CandidateAudioSegment(BaseModel):
    segment_index: int
    segment_name: str
    start_seconds: float
    end_seconds: float
    audio_path: str | None = None
    audio_exists: bool = False


class CandidateScoutRecord(BaseModel):
    candidate_id: str
    rank: int
    status: ScoutStatus
    source_url: str
    youtube_video_id: str | None = None
    title: str
    duration_seconds: int | None = None
    local_video_path: str | None = None
    download_dir: str | None = None
    audio_working_dir: str | None = None
    audio_segments: list[CandidateAudioSegment] = Field(default_factory=list)
    audio_segment_count: int = 0
    extracted_at: datetime | None = None
    failure_reason: str | None = None
    transcript_path: str | None = None
    transcript_char_count: int = 0
    transcript_truncated_for_review: bool = False
    transcript_relevance_score: float | None = None
    source_duplicate_score: float | None = None
    source_duplicate_reason: str | None = None
    content_redundancy_score: float | None = None
    content_redundancy_reason: str | None = None
    extractability_score: float | None = None
    llm_relevance_score: float | None = None
    accepted_for_queue: bool | None = None
    decision: str | None = None
    reason: str | None = None
    supporting_evidence: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    raw_llm_output: str | None = None


class CandidateScoutSnapshot(BaseModel):
    schema_version: int = 1
    generated_at: datetime
    loop_profile_id: str
    download_manifest_path: str | None = None
    download_manifest_generated_at: datetime | None = None
    target_gap: KnowledgeGap
    requested_top_n: int
    selected_candidate_count: int = 0
    audio_extracted_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    records: list[CandidateScoutRecord] = Field(default_factory=list)
    warnings: list[CandidateScoutWarning] = Field(default_factory=list)


class CandidateSelectionRecord(BaseModel):
    candidate_id: str
    rank: int
    accepted_for_queue: bool
    decision: str
    reason: str | None = None
    source_url: str
    youtube_video_id: str | None = None
    title: str
    local_video_path: str | None = None
    transcript_path: str | None = None
    llm_relevance_score: float | None = None
    transcript_relevance_score: float | None = None
    source_duplicate_score: float | None = None
    content_redundancy_score: float | None = None
    extractability_score: float | None = None
    supporting_evidence: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class CandidateSelectionSnapshot(BaseModel):
    schema_version: int = 1
    generated_at: datetime
    loop_profile_id: str
    scout_snapshot_path: str | None = None
    scout_snapshot_generated_at: datetime | None = None
    target_gap: KnowledgeGap
    acceptance_threshold: float
    candidate_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    records: list[CandidateSelectionRecord] = Field(default_factory=list)
