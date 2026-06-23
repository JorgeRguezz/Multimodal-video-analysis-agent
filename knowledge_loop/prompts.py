from __future__ import annotations


SYSTEM_SEARCH_PLANNER = """You are a search-planning component for a gameplay knowledge acquisition loop.

Your job is to propose search queries only. Do not claim that any video exists.
Do not return candidate videos, channels, URLs, transcripts, or rankings.
Return strict JSON only.

NOTE: We are currently in 2026"""


USER_SEARCH_PLAN_TEMPLATE = """Plan targeted video searches for a knowledge acquisition loop.

Game/profile:
- profile_id: {profile_id}
- display_name: {profile_display_name}
- extraction_game_id: {extraction_game_id}

Target gap:
- gap_id: {gap_id}
- gap_kind: {gap_kind}
- topic_key: {topic_key}
- facet: {facet}
- priority: {priority}
- coverage_score: {coverage_score}
- coverage_gap_score: {coverage_gap_score}
- weak_redundancy_score: {weak_redundancy_score}
- missing_facet_score: {missing_facet_score}
- freshness_score: {freshness_score}
- actionability_score: {actionability_score}
- evidence_summary: {evidence_summary}
- reasons: {reasons}

Generate {top_n_intents} search intents that would help find videos likely to fill this exact gap.

Rules:
- Return strict JSON only, with no markdown and no commentary.
- Use the target game implied by the profile and topic key.
- Make queries specific enough to be useful on YouTube or a similar video search backend.
- Prefer educational/explanatory sources when the gap is conceptual.
- Prefer current/recent wording only when useful, especially if freshness is unknown or stale.
- Avoid overly broad searches that only repeat the game name or the entity name.
- Do not invent video titles, channels, URLs, dates, or metadata.
- Do not include duplicate or near-duplicate queries.

Required JSON schema:
{{
  "search_intents": [
    {{
      "query": "search query string",
      "purpose": "why this query is useful for the target gap",
      "source_type": "educational_guide | matchup_breakdown | vod_review | current_patch_or_season_guide | mechanics_guide | other",
      "expected_signal": "what evidence in a result would indicate the video can fill the gap"
    }}
  ]
}}"""


SYSTEM_YOUTUBE_CANDIDATE_RANKER = """
You are ranking YouTube video candidates for a gameplay knowledge acquisition loop.

Reasoning: medium

Use only the provided candidate metadata. Do not invent transcript contents, hidden video contents,
channels, URLs, dates, or performance metrics.

Return strict JSON only.

<|channel|>analysis<|message|>User asks: "What is 2 + 2?" Simple arithmetic. Provide answer.<|end|>
<|start|>assistant<|channel|>final<|message|>2 + 2 = 4.<|return|>
"""


USER_YOUTUBE_CANDIDATE_RANK_TEMPLATE = """Rank YouTube video candidates for the target knowledge gap.

Current date context: 2026.

Target gap:
- gap_id: {gap_id}
- gap_kind: {gap_kind}
- topic_key: {topic_key}
- facet: {facet}
- priority: {priority}
- evidence_summary: {evidence_summary}
- reasons: {reasons}

Search intents used:
{search_intents}

Candidate metadata:
{candidates}

Ranking guidance:
- Prefer candidates whose title and description directly match the target gap.
- Prefer educational/explanatory videos likely to contain reusable gameplay knowledge.
- Prefer newer videos when the topic may be patch/season sensitive.
- Balance freshness against engagement: very new videos may have fewer views/likes.
- Prefer videos in the 3 to 20 minute range for this stage.
- Use views and likes as engagement signals; ignore comments.
- Penalize vague, entertainment-only, montage-only, or off-topic candidates.
- Do not assume transcript content is available yet.

Return strict JSON only, with this schema:
{{
  "ranked_candidates": [
    {{
      "candidate_id": "candidate id from the input",
      "llm_score": 0.0,
      "reason": "short reason grounded in the title/description/metadata"
    }}
  ]
}}

Use llm_score values between 0.0 and 1.0."""


SYSTEM_CANDIDATE_TRANSCRIPT_REVIEWER = """You are reviewing one downloaded gameplay video candidate for a knowledge acquisition loop.

Use only the provided metadata, scores, and transcript. Do not invent visual content, hidden transcript
content, source metadata, or gameplay claims not grounded in the transcript.

Return strict JSON only."""


USER_CANDIDATE_TRANSCRIPT_REVIEW_TEMPLATE = """Review this candidate video for the target knowledge gap.

Current date context: 2026.

Target gap:
- gap_id: {gap_id}
- gap_kind: {gap_kind}
- topic_key: {topic_key}
- facet: {facet}
- priority: {priority}
- evidence_summary: {evidence_summary}
- reasons: {gap_reasons}

Candidate:
- candidate_id: {candidate_id}
- rank: {rank}
- title: {title}
- source_url: {source_url}
- youtube_video_id: {youtube_video_id}
- duration_seconds: {duration_seconds}
- first_pass_final_score: {first_pass_final_score}

Deterministic scout metrics:
- transcript_relevance_score: {transcript_relevance_score}
- source_duplicate_score: {source_duplicate_score}
- source_duplicate_reason: {source_duplicate_reason}
- content_redundancy_score: {content_redundancy_score}
- content_redundancy_reason: {content_redundancy_reason}
- extractability_score: {extractability_score}
- transcript_char_count: {transcript_char_count}
- transcript_truncated_for_review: {transcript_truncated_for_review}

Full transcript:
{transcript}

Review guidance:
- Score whether this candidate is likely to improve the exact target gap if ingested.
- Prefer transcript-grounded educational explanations, route decisions, matchup logic, concepts, or examples.
- Penalize vague entertainment-only content, off-topic transcript content, or content that does not address the target topic/facet.
- If source_duplicate_score is 1.0, keep relevance grounded but note the duplicate risk.
- The decision field should be "accept" when relevance_score >= 0.85 and the candidate is not an exact duplicate; otherwise "reject".

Return strict JSON only, with this schema:
{{
  "candidate_id": "{candidate_id}",
  "relevance_score": 0.0,
  "decision": "accept | reject",
  "reason": "short reason grounded in transcript and metrics",
  "supporting_evidence": [
    "short transcript-grounded evidence"
  ],
  "risks": [
    "short risk or caveat"
  ]
}}

Use relevance_score values between 0.0 and 1.0."""
