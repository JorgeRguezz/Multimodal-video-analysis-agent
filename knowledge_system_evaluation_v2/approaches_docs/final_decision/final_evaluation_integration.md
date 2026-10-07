# Final Evaluation Integration Plan

This document details the practical execution steps for evaluating the Multimodal Graph-RAG system for publication in **IEEE Access**. It translates our defined evaluation goals into a concrete, phase-by-phase data pipeline and methodology.

## Pre-requisites
* The `arcade_forum_qa_test.json` and `mobafire_forum_qa_test.json` datasets exist in `scripts/`.
* The `knowledge_system_evaluation_v2/` folder is ready to house scripts and pipeline outputs.

---

## Phase 1: Data Curation & Legacy Filtering

**Objective:** Standardize the raw data, split it correctly for academic rigor, and identify temporal shifts in the game's meta.

1.  **Merge Datasets:** 
    *   **Action:** Run a script to concatenate Arqade and MobaFire JSON files.
    *   **Output:** `community_qa_dataset.json`
2.  **Legacy Tagging:**
    *   **Action:** Run a lightweight LLM over the dataset. Ask the LLM to identify if the question relies on outdated mechanics (e.g., removed Mythic items, old runes).
    *   **Output:** Each question is tagged with either `[Current]` or `[Legacy]`.
3.  **Video Corpus Registry Compilation:**
    *   **Action:** Script scans the `knowledge_sanitization/cache/` to document the system's ingested videos.
    *   **Output:** `video_corpus_registry.json` containing metadata (video name, focus champions, and crucially, the release year/patch of the video).

---

## Phase 2: Corpus Answerability Labeling

**Objective:** Prevent unfair penalization of the RAG system when the knowledge base lacks required information. Answerability is determined from claim-level corpus evidence, not an uncalibrated cross-encoder score.

1.  **Gold-Anchored Claim Decomposition:**
    *   **Action:** Derive 1-10 essential claims from the Question and Gold Answer. Use the Gold Answer to establish intended scope without requiring exact wording or incidental details.
2.  **High-Recall Hybrid Search:**
    *   **Action:** Search with both the original question and every claim-specific query. Build a pool of up to 50 deduplicated candidates from BM25, dense chunks, entity/graph retrieval, the global graph, and visual support.
    *   **Coverage Rule:** Preserve evidence coverage for claim-specific queries. Broad-question relevance alone must not eliminate all candidates for a narrower claim.
3.  **Evidence-Only Claim Judgment:**
    *   **Action:** Use the local LLM to classify every claim as `supported`, `contradicted`, or `insufficient` using only supplied evidence.
    *   **Citation Rule:** `supported` and `contradicted` require at least one valid evidence ID from the supplied batch. Retry a semantically invalid response once; if the citation remains missing or invalid, downgrade the decision to `insufficient` and flag it for review.
4.  **Context-Bounded Cross-Batch Consolidation:**
    *   **Action:** Screen evidence in small batches and retain both decisive citations and IDs of excerpts that contribute partial information. For unresolved claims, run a final per-claim judgment over a bounded, deduplicated cross-batch shortlist. This permits complementary excerpts to establish a claim without placing all 50 chunks in one prompt.
5.  **Answerability Labels:**
    *   **Full:** All essential claims are supported.
    *   **Partial:** At least one, but not all, essential claims are supported.
    *   **None:** No essential claims are supported.
    *   **Review handling:** `answerability_status` remains `Full`, `Partial`, or `None`. Contradictions, ambiguity, temporal conflict, or judge/schema failures set `requires_review: true` and should be audited before Phase 6.
    *   **Score:** Number of supported essential claims divided by the total number of essential claims.
6.  **Required Audit Artifact:**
    *   **Action:** Run `phase_1_2_3_4_scripts/evaluate_answerability.py` with `--include-details` for official data generation.
    *   **Output:** Retain method/model provenance, source index, run parameters, claims, retrieved evidence, judgments, valid citations, and review reasons in `answerability_details`.
    *   **Rule:** A summary-only file containing only `answerability_status` and `answerability_score` is a smoke-test artifact and cannot be used as the publication-grade answerability dataset.
7.  **Stratified Interpretation:**
    *   **Action:** Slice later metrics by finalized `Full`, `Partial`, and `None` corpus-coverage labels.
    *   **Rule:** A correct-looking answer in the `None` stratum is not automatically a retrieval success. Audit its retrieved evidence: supporting corpus evidence means the answerability label was a false negative; absent evidence means the response came from parametric knowledge or was ungrounded.

---

## Phase 3: Inference & Ablation Pipeline

**Objective:** Run the end-to-end inference across different configurations to scientifically isolate the value of the Multimodal Graph-RAG architecture.

1.  **Baseline Generation:**
    *   **Vanilla Base:** A standard, unprompted LLM chatbot experience (no system instructions, no retrieval context).
    *   **SOTA Base:** A strong frontier LLM answers from internal knowledge without retrieval context or external tools.
    *   **BM25 (Sparse Retrieval):** Run the queries using standard keyword search over the video transcripts.
2.  **System Ablations:**
    *   **ASR-Only RAG:** Retrieval restricted to Whisper transcripts (embedded dynamically).
    *   **Vision-Only RAG:** Retrieval restricted to VLM frame descriptions (embedded dynamically).
    *   **Vector-Only RAG:** Standard dense embedding retrieval (fused ASR + Vision) without graph traversal.
3.  **Full System:**
    *   **Graph-RAG:** The complete, hybrid multimodal system.

The originally planned strict-prompt, empty-context condition is not included in the final seven-system evaluation. The recorded implementation returned the application's deterministic insufficient-evidence fallback before any language-model generation, so it was not a valid prompt-only model result. The raw artifact is preserved for auditability. `evaluated_datasets/community_qa_dataset_final_v2.json` remains an intermediate seven-system inference artifact with incorrect legacy answerability labels; the paper uses `evaluated_datasets/community_qa_dataset_final_v3.json` after claim/evidence relabeling and audit of rows marked `requires_review`.

4.  **Output:** 
    *   A consolidated JSON file mapping `Question -> Configuration -> Generated Answer & Retrieved Chunks`.
    *   *Through this execution, we have successfully recorded exactly how the system behaves with zero context, limited context, semantic context, and full graph context.*

---

## Phase 4: Deterministic Retrieval Evaluation

**Objective:** Prove mathematically that Graph-RAG finds better context than Vector-RAG and BM25 before any LLM generation occurs.

1.  **Entity Extraction:** Extract key entities (Champions, Items, Abilities) from the Gold Answer.
2.  **Metric Calculation:** For each ablation (BM25, Vector, Graph):
    *   **Entity Hit Rate (Recall):** Percentage of Gold Entities present in the retrieved chunks.
    *   **Mean Reciprocal Rank (MRR):** Rank of the first highly relevant chunk.
    *   **Recall@K / Precision@K:** Calculated at K = 5, 10, and 20. 

---

## Phase 5: Generation Evaluation (LLM-as-a-Judge)

**Objective:** Score the final text generated by the system using an automated framework.

1.  **Semantic Overlap:** 
    *   **Action:** Run **BERTScore** between the System's generated answer and the community Gold Answer.
2.  **Reference Answer Correctness:** 
    *   **Action:** Use an LLM Judge (e.g., GPT-5.4-mini at temperature 0) to score the generated answer strictly against the Gold Answer on a 0-2 scale (0=Incorrect, 1=Partial, 2=Substantially correct). The context is explicitly omitted here so the judge evaluates objective correctness, not faithfulness.
3.  **Faithfulness & Answer Relevance:** 
    *   **Action:** Run a structured LLM-as-a-judge prompt to score:
        *   **Answer Relevance:** Does the text address the prompt?
        *   **Faithfulness:** Is the generated answer fully grounded in the retrieved chunks?
4.  **Refusal / Fallback Rate:**
    *   **Action:** Calculate the frequency at which the system emits a low-evidence fallback response, grouped by difficulty strata.
5.  **The Legacy / Currency Distinction (Critical for Publication):**
    *   **Action:** Cross-reference scores with the `[Current]` and `[Legacy]` tags from Phase 1.
    *   **Rule:** If a question is `[Legacy]`, and the system generates an answer that is *wrong for modern gameplay* but *faithful to a legacy video*, it is recorded as a **Success for System Faithfulness**. This explicitly separates *System Accuracy* from *Knowledge Base Currency* in the paper's discussion section.
6.  **Difficulty Stratification:** 
    *   **Action:** Chart the scores across the Phase 2 difficulty strata (`Full`, `Partial`, `None`).

---

## Phase 6: Statistical Significance & Resource Cost

**Objective:** Ensure all reported gains are statistically stable and contextualize them with operational efficiency data.

1.  **Paired Bootstrap Confidence Intervals:** 
    *   **Action:** For every computed metric (Retrieval Recall, Faithfulness, Correctness, etc.), run 10,000 paired bootstrap iterations comparing Graph-RAG vs Vector-Only RAG.
    *   **Output:** The 95% Confidence Interval for the difference in means (e.g., "Graph-RAG improved faithfulness by +0.09 [95% CI: +0.04, +0.14]").
2.  **Efficiency Cost Reporting:**
    *   **Action:** Extract and document metrics from the ingestion logs (mean video processing time, graph nodes/edges per video) and the inference logs (p50/p95 latency).

---

## Phase 7: Qualitative Positive-Control Examples

**Objective:** Demonstrate concretely that the system can answer correctly when the relevant evidence is actually present in the processed video corpus, while avoiding the overclaim that a few examples statistically prove corpus coverage.

1.  **Manual Example Selection:**
    *   **Action:** Select 4-6 representative questions for which the answer is manually verified to exist in the processed videos.
    *   **Required Metadata:** For each example, record the source video, timestamp or segment range, retrieved evidence snippet or short evidence summary, system answer, and a short human assessment.
2.  **Balanced Coverage of System Capabilities:**
    *   **Action:** Choose examples that show different evidence pathways:
        *   ASR-heavy strategy/build evidence.
        *   VLM/visual gameplay-state evidence.
        *   Graph/entity-relation evidence.
        *   Temporal or timestamp-grounded evidence.
        *   At least one justified refusal where the corpus lacks enough evidence.
3.  **Qualitative Table for the Paper:**
    *   **Output:** A compact table with columns such as:
        *   Question.
        *   Source video/time.
        *   Evidence modality or retrieval source.
        *   Retrieved evidence summary.
        *   System answer summary.
        *   Human assessment.
    *   **Rule:** Present these cases as illustrative positive controls, not as a replacement for the quantitative benchmark.
4.  **Interpretation in the Discussion:**
    *   **Rule:** Use cautious wording:
        *   "These examples demonstrate that the pipeline can answer correctly when the necessary evidence is present in the processed corpus."
        *   "Together with the high refusal and low over-refusal rates in the community benchmark, they support the interpretation that corpus coverage is a major practical bottleneck."
    *   **Avoid:** "These examples prove that corpus coverage is the only/main issue."
5.  **Optional Scale-Up Path:**
    *   **Action:** If needed for the final IEEE Access submission, expand the qualitative set into a 30-50 question coverage-controlled benchmark.
    *   **Expanded Output:** Each question should be manually linked to one or more source segments, allowing quantitative reporting of answer correctness, faithfulness, refusal rate, and gold-segment/evidence hit rate.
---

## Phase 8: Corpus-Size Scaling Analysis

**Objective:** Provide direct supporting evidence for the claim that the system is limited by the amount of processed video knowledge available. This phase asks whether the same Graph-RAG pipeline improves when progressively more videos are made available at inference time.

1.  **Subset Selection:**
    *   **Action:** Load `evaluated_datasets/community_qa_dataset_final_v3.json` and select only rows where `ablations.graph_rag.is_refusal == false`.
    *   **Expected Size:** 87 questions.
    *   **Rationale:** This uses the questions that the full 38-video Graph-RAG system actually answered, allowing a fixed answered-question slice to be tested under smaller corpus conditions. This is distinct from the `Full`/`Partial` corpus-answerability subset used elsewhere.
2.  **Corpus-Size Conditions:**
    *   **Action:** Load the existing sanitized cache once and filter `InferenceService.stores` at runtime to deterministic nested video subsets.
    *   **Sizes:** 5, 10, 20, 30, and 38 videos.
    *   **Rule:** Do not rebuild extraction, sanitization, vector stores, or graph caches for this experiment. The scaling intervention is only the number of available loaded video stores.
3.  **Inference Script:**
    *   **Action:** Run `phase_8/run_graph_rag_corpus_scaling.py`.
    *   **Output:** `phase_8/graph_rag_corpus_scaling_results.json`.
    *   **Schema:** The output resembles the final dataset but does not include the multi-system `ablations` object. Instead, each question contains a `covered_videos` object keyed by corpus size. Each run stores selected videos, answer text, evidence sources, evidence contexts, provisional refusal label, and empty metric fields.
4.  **Metric Scoring:**
    *   **Action:** Run `phase_8/evaluate_graph_rag_corpus_scaling_metrics.py` after inference completes.
    *   **Output:** `phase_8/graph_rag_corpus_scaling_results_scored.json`.
    *   **Metrics:**
        *   judged refusal label and refusal rate,
        *   reference correctness using the same 0-2 LLM-judge rubric as Phase 5,
        *   RAGAS faithfulness over retrieved evidence contexts.
    *   **Important:** The inference-time refusal label is heuristic and should not be reported. The paper should report refusal only after the scoring script sets `refusal_method: "judge"`.
5.  **Aggregation:**
    *   **Action:** Run `phase_8/summarize_graph_rag_corpus_scaling.py`.
    *   **Outputs:**
        *   `phase_8/graph_rag_corpus_scaling_summary.json`,
        *   `phase_8/graph_rag_corpus_scaling_summary.csv`.
    *   **Reported Fields:** number of videos, number of answered/scored questions, refusal rate, mean correctness, mean faithfulness, and mean final evidence count.
6.  **Paper Interpretation:**
    *   **Rule:** Present this phase as a supporting scaling analysis, not as the primary benchmark. The intended interpretation is that limited corpus size is a practical bottleneck: as more processed videos are available, the fixed Graph-RAG pipeline should abstain less and tend to achieve higher correctness. If the trend is non-monotonic, discuss retrieval noise rather than hiding it.

