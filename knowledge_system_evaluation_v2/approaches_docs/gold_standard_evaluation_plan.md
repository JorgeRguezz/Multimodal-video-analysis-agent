# Gold Standard Evaluation Plan for the Multimodal Gameplay Knowledge System

## 1. Purpose

This document defines a full, publication-grade evaluation plan for the multimodal gameplay knowledge system. The goal is to move from a final-answer-only evaluation to a controlled, evidence-grounded evaluation that can show:

1. Whether the explicit knowledge graph improves gameplay question answering.
2. Whether automatic graph construction is good enough to support that benefit.
3. Which pipeline components contribute to answer quality, grounding, retrieval quality, and efficiency.
4. Where system failures originate: video extraction, graph construction, retrieval, generation, or abstention.

The evaluation should be designed around the central research question:

> Does adding a provenance-aware knowledge graph improve grounded gameplay QA compared with parametric LLMs and vector-only RAG, especially for temporal, entity-relation, visual, and cross-video questions?

## 2. Evaluation Claims

The revised paper should avoid relying on one global leaderboard. It should make several separable claims and evaluate each one directly.

| Claim | Required Evidence |
|---|---|
| The system can extract useful multimodal evidence from gameplay video. | Champion recognition accuracy, ASR quality, event coverage, VLM hallucination rate, segment-summary quality. |
| The automatic graph contains useful gameplay knowledge. | Node/edge precision, recall, F1, entity-linking accuracy, source provenance accuracy. |
| The graph improves retrieval beyond vector-only RAG. | Evidence Recall@k, MRR, nDCG@k, timestamp hit rate, relation/path retrieval success. |
| The graph improves final QA. | Claim precision/recall/F1, hallucinated claim rate, citation accuracy, human usefulness ratings. |
| Manual or curated graph quality creates a higher ceiling. | Manual-KG-RAG vs auto-KG-RAG vs vector-only RAG. |
| The system is practical under local hardware constraints. | Preprocessing time, query latency, storage, GPU memory, build time per video. |

## 3. Required Evaluation Assets

### 3.1 Evaluation Corpus Registry

Create a registry describing every video used in the evaluation.

Suggested file:

```text
knowledge_system_evaluation_v2/data/video_corpus_registry.json
```

Required fields:

```json
{
  "video_id": "video_001",
  "video_name": "3_Minute_Aatrox_Guide_-_A_Guide_for_League_of_Legends",
  "source_path": "/absolute/path/to/video.mp4",
  "source_url": "optional",
  "duration_seconds": 180,
  "champions": ["AATROX"],
  "content_type": "guide | gameplay_analysis | matchup | lore | general_advice",
  "language": "en",
  "patch_or_date": "optional",
  "included_in_split": "train_dev | evaluation",
  "notes": ""
}
```

What needs to be developed:

- A script that scans processed caches and creates an initial registry.
- A manual review pass to verify video names, duration, champion focus, source URL, and content type.
- A stable video ID mapping so evaluation data do not depend on long filesystem names.

Suggested script:

```text
knowledge_system_evaluation_v2/scripts/build_video_corpus_registry.py
```

Access required:

- Processed sanitized build caches.
- Original or accessible video files for duration checks and annotation.
- Optional source URLs for reproducibility and citation.

### 3.2 Gold Segment And Frame Annotations

The evaluation needs manually annotated evidence windows. These annotations are the backbone for retrieval, timestamp, grounding, and answer evaluation.

Suggested file:

```text
knowledge_system_evaluation_v2/data/gold_evidence_windows.json
```

Required fields:

```json
{
  "evidence_id": "ev_0001",
  "video_id": "video_001",
  "video_name": "3_Minute_Aatrox_Guide_-_A_Guide_for_League_of_Legends",
  "time_start": 30.0,
  "time_end": 60.0,
  "segment_ids": ["3_Minute_Aatrox_Guide_-_A_Guide_for_League_of_Legends_1"],
  "frame_ids": ["1_0", "1_1", "1_2"],
  "visible_entities": ["AATROX", "Fiora"],
  "spoken_entities": ["Conqueror", "Black Cleaver"],
  "events": [
    {
      "event_type": "itemization_advice",
      "description": "The guide recommends building bruiser items for sustained fights."
    }
  ],
  "gold_claims": [
    "Aatrox benefits from Conqueror in extended top-lane trades."
  ],
  "gold_triples": [
    ["AATROX", "USES_RUNE", "CONQUEROR"],
    ["CONQUEROR", "SUPPORTS", "EXTENDED_TRADES"]
  ],
  "annotation_confidence": "high | medium | low",
  "annotator_ids": ["ann_1", "ann_2"]
}
```

What needs to be developed:

- An annotation guideline describing what counts as an event, entity, relation, and support claim.
- A small annotation UI or spreadsheet workflow for selecting time windows and writing claims.
- A validator that checks times, video IDs, duplicate evidence IDs, and segment references.

Suggested scripts:

```text
knowledge_system_evaluation_v2/scripts/validate_gold_evidence.py
knowledge_system_evaluation_v2/scripts/export_annotation_spreadsheet.py
knowledge_system_evaluation_v2/scripts/import_annotation_spreadsheet.py
```

Access required:

- Original videos or frame previews.
- Existing system segment timestamps.
- At least two annotators with League of Legends knowledge for a subset of the data.

Recommended annotation scale:

- 12-16 videos deeply annotated.
- 250-400 evidence windows.
- Each selected video should include early, middle, and late windows where available.
- Include windows with visual-only evidence, audio-only evidence, and combined evidence.

### 3.3 Gold QA Dataset

The QA benchmark must be grounded in the gold evidence windows.

Suggested file:

```text
knowledge_system_evaluation_v2/data/gold_qa_dataset.json
```

Required fields:

```json
{
  "question_id": "q_0001",
  "question": "What rune does the Aatrox guide recommend for extended trades, and why?",
  "query_type": "entity_relation",
  "required_reasoning": ["entity_lookup", "relation_lookup"],
  "expected_answer_claims": [
    "The guide recommends Conqueror for Aatrox.",
    "Conqueror is useful because Aatrox benefits from extended trades."
  ],
  "gold_evidence_ids": ["ev_0001"],
  "expected_videos": ["video_001"],
  "expected_entities": ["AATROX", "CONQUEROR"],
  "expected_relations": [["AATROX", "USES_RUNE", "CONQUEROR"]],
  "answerability": "answerable",
  "difficulty": "easy | medium | hard",
  "notes": ""
}
```

Required query types:

| Query Type | Purpose |
|---|---|
| `entity_fact` | Tests direct retrieval of champion, item, rune, objective, or concept facts. |
| `entity_relation` | Tests graph-style relation retrieval, e.g. champion-item, champion-matchup, action-outcome. |
| `temporal_event` | Tests time-localized gameplay understanding. |
| `sequence_summary` | Tests multi-step event understanding inside one clip or segment range. |
| `visual_detail` | Tests whether frame/VLM-derived evidence helps answer screen-state questions. |
| `audio_advice` | Tests transcript-driven guide/advice retrieval. |
| `cross_video_comparison` | Tests global graph and multi-video evidence retrieval. |
| `unanswerable` | Tests abstention and hallucination resistance. |

What needs to be developed:

- A dataset builder that links QA items to evidence IDs.
- A validator that verifies every question has valid evidence references unless marked unanswerable.
- A script that produces balanced train/dev/evaluation splits if parameter tuning is needed.

Suggested scripts:

```text
knowledge_system_evaluation_v2/scripts/validate_gold_qa_dataset.py
knowledge_system_evaluation_v2/scripts/summarize_qa_dataset.py
knowledge_system_evaluation_v2/scripts/create_eval_splits.py
```

Access required:

- Gold evidence windows.
- Annotators who can write questions and expected claim-level answers.

Recommended scale:

- 300-500 total questions.
- 40-70 questions per main query type.
- 10-15% unanswerable questions.
- At least 80 temporal questions.
- At least 50 cross-video or comparison questions if the global graph is a major claim.

## 4. Manual Knowledge Graph Ceiling

The most important addition for the revised paper is a manual or curated KG baseline. This isolates the value of graph-based retrieval from the quality of automatic graph extraction.

### 4.1 Manual KG Scope

The manual KG does not need to cover every processed video. It should cover the deeply annotated evaluation subset.

Suggested file:

```text
knowledge_system_evaluation_v2/data/manual_gold_kg.json
```

Suggested schema:

```json
{
  "nodes": [
    {
      "node_id": "AATROX",
      "type": "CHAMPION",
      "aliases": ["Aatrox"],
      "description": "Top-lane bruiser champion."
    }
  ],
  "edges": [
    {
      "edge_id": "edge_0001",
      "source": "AATROX",
      "relation": "USES_RUNE",
      "target": "CONQUEROR",
      "description": "The guide recommends Conqueror for extended trades.",
      "evidence_ids": ["ev_0001"],
      "confidence": "high"
    }
  ]
}
```

Suggested node types:

- `CHAMPION`
- `ITEM`
- `RUNE`
- `ABILITY`
- `OBJECTIVE`
- `ROLE`
- `GAME_PHASE`
- `MATCHUP`
- `EVENT`
- `TACTIC`
- `OUTCOME`
- `MAP_LOCATION`
- `RESOURCE`
- `UNKNOWN`

Suggested relation types:

- `USES_RUNE`
- `BUILDS_ITEM`
- `COUNTERS`
- `SYNERGIZES_WITH`
- `WEAK_AGAINST`
- `STRONG_AGAINST`
- `USES_ABILITY`
- `COMBOS_WITH`
- `OCCURS_BEFORE`
- `CAUSES`
- `LEADS_TO`
- `RECOMMENDS`
- `AVOIDS`
- `ROAMS_TO`
- `CONTESTS_OBJECTIVE`
- `POSITIONED_AT`
- `HAS_OUTCOME`
- `SUPPORTED_BY_EVIDENCE`

What needs to be developed:

- A manual KG authoring format.
- A converter from manual KG JSON to the same retrieval format used by the inference system.
- A retrieval runner that can use the manual KG instead of the automatically extracted graph.
- A graph validator that checks node references, relation labels, duplicate edges, and evidence provenance.

Suggested scripts:

```text
knowledge_system_evaluation_v2/scripts/validate_manual_gold_kg.py
knowledge_system_evaluation_v2/scripts/build_manual_kg_cache.py
knowledge_system_evaluation_v2/scripts/run_manual_kg_rag.py
```

Access required:

- Gold evidence windows.
- Domain annotators.
- Ability to run the inference system against an alternate graph cache.

Expected result:

- If manual-KG-RAG beats vector-only RAG, this shows that graph structure is useful.
- If auto-KG-RAG approaches manual-KG-RAG, this shows automatic graph extraction is effective.
- If manual-KG-RAG helps but auto-KG-RAG does not, the paper can honestly identify graph extraction as the bottleneck.

## 5. Systems And Baselines

All systems should answer the same QA dataset with identical generation settings where possible.

### 5.1 Required Systems

| System ID | Description | Purpose |
|---|---|---|
| `llm_local_no_context` | Local GPT-OSS or equivalent without retrieval. | Measures local parametric knowledge. |
| `llm_strong_no_context` | Strong proprietary or frontier LLM without retrieval. | Measures strong general League knowledge. |
| `transcript_rag` | RAG over ASR transcript only. | Tests audio-only contribution. |
| `summary_vector_rag` | Dense retrieval over segment summaries/chunks only. | Main vector-only RAG baseline. |
| `frame_text_rag` | Retrieval over frame VLM outputs plus transcript. | Tests visual-description contribution without KG. |
| `auto_local_graph_rag` | Per-video automatic graph retrieval only. | Tests local graph utility. |
| `auto_global_graph_rag` | Local + global automatic graph retrieval. | Tests cross-video/global graph utility. |
| `full_system` | Current full hybrid system. | Main system. |
| `manual_kg_rag` | RAG using manually curated KG for the annotated subset. | Upper bound for KG usefulness. |

What needs to be developed:

- A unified experiment runner that can switch retrieval modes.
- A common output schema for all systems.
- Configuration snapshots so every run is reproducible.

Suggested output file:

```text
knowledge_system_evaluation_v2/results/system_outputs/{run_id}.jsonl
```

Suggested output row:

```json
{
  "run_id": "2026_kg_eval_full_system",
  "system_id": "full_system",
  "question_id": "q_0001",
  "question": "...",
  "answer": "...",
  "retrieved_evidence": [
    {
      "video_id": "video_001",
      "chunk_id": "chunk-...",
      "time_span": "0:30-1:00",
      "source": "dense_chunk|entity_graph",
      "score": 0.62,
      "text": "..."
    }
  ],
  "latency_seconds": 8.2,
  "debug": {}
}
```

Suggested scripts:

```text
knowledge_system_evaluation_v2/scripts/run_system_outputs.py
knowledge_system_evaluation_v2/scripts/run_all_baselines.py
knowledge_system_evaluation_v2/scripts/check_run_completeness.py
```

Access required:

- Local model runtimes.
- Optional API access for the strong LLM baseline.
- Existing sanitized caches.
- Manual KG cache for the manual-KG condition.

## 6. Ablation Studies

Ablations should test exactly which components matter.

| Ablation | Comparison | What It Tests |
|---|---|---|
| Remove graph retrieval | `full_system` vs dense chunk retrieval only | Whether the graph improves evidence selection. |
| Local graph only | `auto_local_graph_rag` vs `auto_global_graph_rag` | Whether global graph merging helps. |
| Remove entity VDB | Full retrieval vs no entity-vector branch | Whether entity embeddings help seed graph retrieval. |
| Remove global graph | Full retrieval vs no global graph branch | Whether cross-video retrieval depends on global graph. |
| Remove visual support | Full retrieval vs no frame-text retrieval | Whether frame VLM outputs help visual queries. |
| Remove ASR | Full extraction/build with captions only | Whether speech contributes to advice questions. |
| Remove HUD champion grounding | Full extraction vs VLM without detected champion context | Whether explicit champion grounding improves descriptions. |
| Remove previous-frame memory | Current VLM prompting vs no prior-frame description | Whether temporal chaining helps event continuity. |
| Sanitization disabled | Sanitized cache vs unsanitized build cache | Whether sanitization improves graph/retrieval quality. |
| Sampling density | 5 frames/30s vs denser sampling subset | Whether sparse sampling misses important gameplay events. |
| Verifier enabled | Current generation vs claim verifier enabled | Whether verifier improves hallucination/abstention tradeoff. |

What needs to be developed:

- Feature flags for retrieval branches.
- Alternate extraction/build runs for expensive ablations.
- A smaller controlled subset for high-cost video reprocessing ablations.
- A configuration manifest per ablation.

Suggested files:

```text
knowledge_system_evaluation_v2/configs/ablation_full_system.json
knowledge_system_evaluation_v2/configs/ablation_vector_only.json
knowledge_system_evaluation_v2/configs/ablation_no_global_graph.json
knowledge_system_evaluation_v2/configs/ablation_manual_kg.json
```

Suggested scripts:

```text
knowledge_system_evaluation_v2/scripts/run_ablation.py
knowledge_system_evaluation_v2/scripts/compare_ablation_results.py
```

Access required:

- Ability to run inference under modified retrieval configurations.
- For extraction ablations, access to original videos and enough GPU time to rebuild caches.

## 7. Component-Level Evaluation

### 7.1 Champion Recognition Evaluation

Purpose:

Evaluate whether HUD-centric champion detection correctly identifies the player champion and teammates.

Gold data needed:

```text
knowledge_system_evaluation_v2/data/gold_champion_frames.json
```

Schema:

```json
{
  "frame_id": "video_001_1_0",
  "video_id": "video_001",
  "frame_path": "...",
  "segment_idx": "1",
  "frame_idx": 0,
  "gold_main_champion": "AATROX",
  "gold_teammates": ["AHRI", "LUX", "LEE SIN", "JINX"],
  "visibility": "clear | partial | unclear"
}
```

Metrics:

- Main champion accuracy.
- Teammate micro/macro precision, recall, F1.
- Unknown/failed detection rate.
- Accuracy by visibility level.

Scripts to develop:

```text
knowledge_system_evaluation_v2/scripts/evaluate_champion_recognition.py
```

Access required:

- Sampled frame images.
- Annotators who can identify LoL champions from HUD portraits.

### 7.2 ASR Evaluation

Purpose:

Measure transcript quality on spoken guide/commentary content.

Gold data needed:

```text
knowledge_system_evaluation_v2/data/gold_transcripts.json
```

Metrics:

- Word Error Rate.
- Named entity error rate for champions, items, runes, abilities.
- Timestamp alignment tolerance, if manually annotated.

Scripts to develop:

```text
knowledge_system_evaluation_v2/scripts/evaluate_asr.py
```

Access required:

- Audio clips.
- Manual transcripts for a subset of clips.

### 7.3 VLM Description Evaluation

Purpose:

Evaluate whether frame-level descriptions correctly capture gameplay state.

Gold data needed:

```text
knowledge_system_evaluation_v2/data/gold_frame_events.json
```

Metrics:

- Entity mention precision/recall.
- Event coverage.
- Hallucinated event rate.
- Visual detail correctness.
- Temporal continuity score for consecutive frames.

Scripts to develop:

```text
knowledge_system_evaluation_v2/scripts/evaluate_vlm_descriptions.py
```

Access required:

- Sampled frames.
- Frame-level system VLM outputs.
- Human annotations of visible events and UI facts.

### 7.4 Segment Summary Evaluation

Purpose:

Evaluate whether segment summaries preserve the important information from frame descriptions and transcript.

Metrics:

- Gold claim recall.
- Unsupported claim rate.
- Redundancy rate.
- Entity preservation rate.

Scripts to develop:

```text
knowledge_system_evaluation_v2/scripts/evaluate_segment_summaries.py
```

Access required:

- Segment summaries.
- Gold evidence windows and claims.

### 7.5 KG Construction Evaluation

Purpose:

Evaluate automatic graph nodes and edges against the manual gold KG.

Metrics:

- Node precision, recall, F1.
- Edge precision, recall, F1.
- Entity type accuracy.
- Relation type accuracy.
- Entity-linking/canonicalization accuracy.
- Provenance accuracy: whether node/edge source IDs point to valid supporting chunks.
- Graph contamination rate: prompt leakage, malformed nodes, placeholder nodes.

Scripts to develop:

```text
knowledge_system_evaluation_v2/scripts/evaluate_kg_construction.py
```

Access required:

- Automatic graph files.
- Manual KG for the same video subset.
- Entity/relation normalization rules.

## 8. Retrieval Evaluation

Retrieval must be evaluated before answer generation. This is where the graph contribution should become visible.

Gold source:

```text
knowledge_system_evaluation_v2/data/gold_qa_dataset.json
knowledge_system_evaluation_v2/data/gold_evidence_windows.json
```

Metrics:

| Metric | Meaning |
|---|---|
| Evidence Recall@k | Fraction of questions where at least one gold evidence item is retrieved in top k. |
| Evidence Precision@k | Fraction of top-k retrieved items that overlap gold evidence. |
| MRR | Rank of the first relevant evidence item. |
| nDCG@k | Ranking quality when multiple evidence items exist. |
| Timestamp Hit Rate | Retrieved evidence overlaps the gold time window within a tolerance. |
| Entity Coverage | Retrieved evidence contains required entities. |
| Relation Coverage | Retrieved evidence supports required gold relations. |
| Cross-Video Coverage | For comparison questions, evidence from all required videos is retrieved. |

Timestamp tolerance:

- Strict: overlap with gold window.
- Moderate: within +/-15 seconds.
- Loose: within +/-30 seconds.

Scripts to develop:

```text
knowledge_system_evaluation_v2/scripts/evaluate_retrieval.py
knowledge_system_evaluation_v2/scripts/plot_retrieval_by_query_type.py
```

Access required:

- System output files containing retrieved evidence.
- Gold evidence windows.
- Mapping from chunks to segments and time windows.

Expected tables:

- Retrieval metrics by system.
- Retrieval metrics by query type.
- Retrieval metrics by answerability.
- Dense-only vs graph-enhanced retrieval comparison.

## 9. Answer-Level Evaluation

Final answers should be scored at claim level.

### 9.1 Automatic Claim-Level Scoring

Convert generated answers into atomic claims and compare them to:

- expected answer claims,
- retrieved evidence,
- gold evidence.

Metrics:

- Claim precision.
- Claim recall.
- Claim F1.
- Unsupported claim rate.
- Contradicted claim rate.
- Abstention rate.
- Correct abstention rate on unanswerable questions.
- Over-abstention rate on answerable questions.
- Citation accuracy.

Suggested intermediate file:

```text
knowledge_system_evaluation_v2/results/claim_judgments/{run_id}.jsonl
```

Schema:

```json
{
  "question_id": "q_0001",
  "system_id": "full_system",
  "answer_claims": [
    {
      "claim": "The guide recommends Conqueror for Aatrox.",
      "label_against_gold": "supported",
      "label_against_retrieved_context": "supported",
      "supporting_evidence_ids": ["ev_0001"],
      "notes": ""
    }
  ],
  "claim_precision": 0.8,
  "claim_recall": 0.67,
  "claim_f1": 0.73
}
```

Scripts to develop:

```text
knowledge_system_evaluation_v2/scripts/extract_answer_claims.py
knowledge_system_evaluation_v2/scripts/judge_answer_claims.py
knowledge_system_evaluation_v2/scripts/evaluate_answers.py
```

Access required:

- Strong LLM judge access or a carefully validated local judge.
- Human-reviewed calibration subset.

Important requirement:

LLM-as-judge scores should be calibrated against human labels on a subset. The paper should report agreement between automated and human judgment before relying on judge scores.

### 9.2 Human Answer Evaluation

Human evaluation should be blind: annotators should not know which system produced each answer.

Systems to include:

- `summary_vector_rag`
- `full_system`
- `manual_kg_rag`
- `llm_strong_no_context`

Recommended number:

- 80-120 questions.
- Balanced across query types.
- Each answer rated by at least two annotators.

Rating dimensions:

| Dimension | Scale |
|---|---|
| Correctness | 1-5 |
| Grounding in provided evidence | 1-5 |
| Usefulness/actionability | 1-5 |
| Completeness | 1-5 |
| Citation/time accuracy | 1-5 |

Pairwise preference can also be collected:

```json
{
  "question_id": "q_0001",
  "answer_a_system_hidden": "system_x",
  "answer_b_system_hidden": "system_y",
  "preferred_answer": "A | B | tie",
  "reason": "more grounded | more useful | more complete | fewer hallucinations"
}
```

Scripts to develop:

```text
knowledge_system_evaluation_v2/scripts/export_human_eval_batches.py
knowledge_system_evaluation_v2/scripts/import_human_eval_results.py
knowledge_system_evaluation_v2/scripts/analyze_human_eval.py
```

Access required:

- Two or more annotators with LoL knowledge.
- Annotation platform, spreadsheet, or lightweight local web UI.

Report:

- Mean score per dimension and system.
- Pairwise win rate.
- Inter-annotator agreement, e.g. Cohen's kappa, Krippendorff's alpha, or weighted agreement.

## 10. Efficiency And Scalability Evaluation

The paper claims practical deployability on moderate hardware, so the evaluation should measure system cost.

Metrics:

| Stage | Metrics |
|---|---|
| Extraction | seconds per video minute, GPU memory, frames processed, ASR time, VLM time, summarization time. |
| Sanitization | seconds per video, records cleaned, records dropped, contamination hits. |
| Build | chunks created, graph nodes/edges, build time, embedding time, graph extraction time. |
| Inference | p50/p95 latency, retrieval time, reranking time, generation time, total time. |
| Storage | cache size per video, vector DB size, graph size. |

Suggested files:

```text
knowledge_system_evaluation_v2/results/efficiency/{run_id}.json
knowledge_system_evaluation_v2/results/efficiency/stage_timings.csv
```

Scripts to develop:

```text
knowledge_system_evaluation_v2/scripts/collect_cache_statistics.py
knowledge_system_evaluation_v2/scripts/profile_inference_latency.py
knowledge_system_evaluation_v2/scripts/summarize_efficiency.py
```

Access required:

- Runtime logs or instrumentation.
- Hardware metadata.
- Repeated query runs for stable latency estimates.

Recommended reporting:

- Query latency p50, p90, p95.
- Preprocessing time per hour of video.
- Storage growth per hour of video.
- Graph growth per video.

## 11. Robustness Evaluation

Robustness tests should evaluate whether the system remains reliable under realistic user and data variation.

### 11.1 Query Robustness

Create paraphrases for a subset of questions.

Metrics:

- Answer consistency.
- Evidence consistency.
- Retrieval overlap between paraphrases.
- Claim-level answer variance.

Script:

```text
knowledge_system_evaluation_v2/scripts/evaluate_query_robustness.py
```

### 11.2 Entity Alias Robustness

Test known spelling variants, abbreviations, and recognition errors.

Examples:

- `pike` vs `Pyke`
- `smoulder` vs `Smolder`
- item/rune aliases

Metrics:

- Retrieval recall under alias perturbation.
- Answer correctness under alias perturbation.

Script:

```text
knowledge_system_evaluation_v2/scripts/evaluate_alias_robustness.py
```

### 11.3 Unanswerable And Adversarial Questions

Include questions that are plausible but unsupported by the processed corpus.

Metrics:

- Correct abstention rate.
- Hallucinated answer rate.
- Overconfident unsupported answer rate.

Script:

```text
knowledge_system_evaluation_v2/scripts/evaluate_unanswerable_questions.py
```

## 12. Statistical Analysis

Use paired tests because each system answers the same questions.

Required analysis:

- Mean and 95% bootstrap confidence intervals.
- Per-query-type breakdown.
- Paired bootstrap significance tests for metric differences.
- Wilcoxon signed-rank tests for human rating dimensions.
- Effect sizes.

Suggested script:

```text
knowledge_system_evaluation_v2/scripts/statistical_analysis.py
```

Expected output:

```text
knowledge_system_evaluation_v2/results/statistics/{run_id}_stats.md
knowledge_system_evaluation_v2/results/statistics/{run_id}_stats.json
```

## 13. Expected Publication Tables

### Table 1: Corpus And Artifact Statistics

Columns:

- Videos.
- Total duration.
- Segments.
- Frames.
- Chunks.
- Graph nodes.
- Graph edges.
- Average nodes/video.
- Average edges/video.

### Table 2: QA Dataset Composition

Columns:

- Query type.
- Number of questions.
- Average evidence windows/question.
- Answerable/unanswerable count.
- Difficulty distribution.

### Table 3: Component Quality

Columns:

- Champion accuracy.
- ASR WER.
- VLM event recall.
- VLM hallucination rate.
- Segment claim recall.
- KG node F1.
- KG edge F1.
- Provenance accuracy.

### Table 4: Retrieval Performance

Rows:

- transcript RAG.
- vector-only RAG.
- auto local Graph-RAG.
- auto global Graph-RAG.
- full system.
- manual KG-RAG.

Columns:

- Recall@5.
- Recall@10.
- MRR.
- nDCG@10.
- timestamp hit@15s.
- relation coverage.

### Table 5: Answer Performance

Columns:

- Claim precision.
- Claim recall.
- Claim F1.
- hallucination rate.
- correct abstention.
- over-abstention.
- citation accuracy.

### Table 6: Ablation Results

Rows:

- full system.
- no graph retrieval.
- no global graph.
- no visual support.
- no ASR.
- no sanitization.
- manual KG.

Columns:

- Retrieval Recall@10.
- Claim F1.
- Hallucination rate.
- Latency.

### Table 7: Human Evaluation

Columns:

- correctness.
- grounding.
- usefulness.
- completeness.
- citation accuracy.
- pairwise win rate.

### Table 8: Efficiency

Columns:

- extraction time/video minute.
- build time/video.
- storage/video.
- p50 latency.
- p95 latency.
- GPU memory.

## 14. Required Repository Structure

Recommended final structure:

```text
knowledge_system_evaluation_v2/
  gold_standard_evaluation_plan.md
  data/
    video_corpus_registry.json
    gold_evidence_windows.json
    gold_qa_dataset.json
    gold_champion_frames.json
    gold_transcripts.json
    gold_frame_events.json
    manual_gold_kg.json
  configs/
    baseline_full_system.json
    baseline_vector_only.json
    baseline_transcript_rag.json
    baseline_manual_kg.json
    ablation_no_global_graph.json
    ablation_no_visual_support.json
  scripts/
    build_video_corpus_registry.py
    validate_gold_evidence.py
    validate_gold_qa_dataset.py
    validate_manual_gold_kg.py
    build_manual_kg_cache.py
    run_system_outputs.py
    run_all_baselines.py
    run_ablation.py
    evaluate_champion_recognition.py
    evaluate_asr.py
    evaluate_vlm_descriptions.py
    evaluate_segment_summaries.py
    evaluate_kg_construction.py
    evaluate_retrieval.py
    extract_answer_claims.py
    judge_answer_claims.py
    evaluate_answers.py
    export_human_eval_batches.py
    import_human_eval_results.py
    analyze_human_eval.py
    collect_cache_statistics.py
    profile_inference_latency.py
    statistical_analysis.py
  results/
    system_outputs/
    retrieval/
    answer_quality/
    human_eval/
    efficiency/
    statistics/
  reports/
    evaluation_summary.md
    tables/
    figures/
```

## 15. Implementation Order

The full gold standard evaluation is large. The implementation should be staged so each stage produces useful paper evidence.

### Stage 1: Dataset And Retrieval Ground Truth

Deliverables:

- `video_corpus_registry.json`
- `gold_evidence_windows.json`
- `gold_qa_dataset.json`
- validators
- retrieval evaluator

Why first:

Without gold evidence, the evaluation cannot prove whether retrieval or graph reasoning works.

### Stage 2: Baseline Runner

Deliverables:

- unified system-output schema
- vector-only baseline
- full-system baseline
- no-context LLM baselines
- retrieval metrics

Why second:

This immediately tests the main reviewer concern: graph retrieval vs vector-only retrieval.

### Stage 3: Manual KG Ceiling

Deliverables:

- `manual_gold_kg.json`
- manual KG cache builder
- manual-KG-RAG output
- comparison against auto-KG and vector-only RAG

Why third:

This directly separates graph usefulness from automatic extraction quality.

### Stage 4: Claim-Level Answer Evaluation

Deliverables:

- claim extraction
- claim judging
- answer metrics
- hallucination and abstention analysis

Why fourth:

Final-answer metrics should be grounded in evidence and claims, not only lexical similarity.

### Stage 5: Component And Ablation Evaluation

Deliverables:

- KG construction evaluation
- champion recognition evaluation
- VLM/ASR subset evaluation
- retrieval and extraction ablations

Why fifth:

This explains why the system succeeds or fails.

### Stage 6: Human Evaluation And Statistics

Deliverables:

- blind human evaluation batches
- inter-annotator agreement
- bootstrap confidence intervals
- final paper tables

Why sixth:

Human evaluation validates automatic metrics and strengthens publication credibility.

## 16. Success Criteria

The revised evaluation should be considered strong if it can answer these questions clearly:

1. Does full Graph-RAG retrieve correct evidence more often than vector-only RAG?
2. Does graph retrieval improve relation, temporal, and cross-video questions specifically?
3. Does manual-KG-RAG outperform vector-only RAG?
4. How close is auto-KG-RAG to manual-KG-RAG?
5. Are final answers more faithful, less hallucinated, or more useful?
6. Does the system abstain correctly when evidence is missing?
7. Where are the main bottlenecks: extraction, graph construction, retrieval, or generation?
8. Is the system practical to run locally at the reported scale?

## 17. External Evaluation Ideas To Cite

The paper can position this evaluation using related evaluation ideas:

- RAGAS-style decomposition of RAG quality into faithfulness, answer relevance, and context use.
- RAGChecker-style fine-grained diagnosis of retriever and generator failures.
- FActScore-style atomic factual claim checking for long-form answers.
- Multi-hop RAG benchmarks that require supporting evidence annotations.
- GraphRAG evaluations that compare graph-based retrieval with conventional vector RAG.
- Video QA benchmarks such as Video-MME and gameplay-focused VLM benchmarks for video-game visual understanding.

These should be used as methodological inspiration, while the actual benchmark should remain domain-specific and evidence-grounded in the processed gameplay videos.
