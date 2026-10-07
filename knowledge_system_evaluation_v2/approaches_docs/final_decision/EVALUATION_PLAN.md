# Gameplay Knowledge System Evaluation Plan

## 1. Goal

This document defines a practical, high-fidelity evaluation for the multimodal gameplay knowledge system. 

The evaluation moves away from internally hand-crafted or video-event-specific questions and instead uses real player questions from external sources. The main goal is to evaluate whether the system can answer the kinds of questions that League of Legends players actually ask, while robustly proving the value of the Knowledge Graph compared to standard Vector RAG.

This plan integrates the practical constraints of the *Approachable* plan with the architectural rigor of the *Gold Standard* plan, minimizing manual annotation work while maximizing defensive, publication-grade metrics.

## 2. Core Evaluation Framing

The central evaluation question is:

> Can a video-derived Graph-RAG knowledge system answer real League of Legends player questions accurately and usefully, while remaining grounded in its processed knowledge corpus?

To prove this without bias, the evaluation relies on **deterministic retrieval metrics** and **established LLM-as-a-Judge frameworks** (like RAGAS) rather than self-graded calibration or flawed string-matching metrics.

## 3. Data Sources

We utilize real community questions and their top/accepted answers.

*   **Arqade (Gaming Stack Exchange):** Sourced via API. Questions are strictly filtered to those containing an accepted community answer with a score > 1 to ensure quality. Located at: `knowledge_system_evaluation_v2/scripts/arcade_forum_qa_test.json`.
*   **MOBAFire:** Sourced via web scraping. Contains practical, meta-heavy League of Legends guide content. Located at: `knowledge_system_evaluation_v2/scripts/mobafire_forum_qa_test.json`.

*(Note: Context regarding the system architecture and generation phases can be found in the `@docs/**` folder).*

## 4. Phase-by-Phase Execution

### Phase 1: Data Curation & Legacy Filtering
League of Legends is a frequently updated game. To ensure fairness, we must identify questions (and videos) that rely on outdated mechanics.

1.  **Combine Datasets:** Merge the Arqade and MobaFire JSON outputs into a unified `community_qa_dataset.json`.
2.  **Legacy Tagging:** Run an LLM script over the dataset to tag questions as `[Current]` or `[Legacy]` based on mentions of removed items (e.g., Mythics), old runes, or reworked champions.
3.  **Video Corpus Registry:** Create a `video_corpus_registry.json` scanning the processed system caches to document what the system knows (video names, focus champions, content type, and the year/patch of the video).

### Phase 2: Corpus Answerability Labeling
We cannot penalize the system for failing to answer a question if the answer does not exist in the processed videos.

1.  **Gold-Anchored Claim Decomposition:** Use the question and Gold Answer to derive 1-10 essential, semantically matchable claims. The Gold Answer defines the intended scope; exact wording and incidental details are not required.
2.  **High-Recall Evidence Search:** Retrieve a pool of up to 50 chunks using the original question and claim-specific searches across BM25, dense chunk retrieval, entity/graph retrieval, global-graph retrieval, and visual support. Retrieval must preserve coverage of the claim-specific searches rather than allowing only the broad question to determine the final pool.
3.  **Evidence-Only Judgment:** An LLM judge evaluates each claim using only supplied excerpts. A `supported` or `contradicted` decision is valid only when it contains at least one valid evidence ID from the supplied batch. Missing or invalid citations trigger one constrained retry; if still invalid, the decision becomes `insufficient` and is flagged for review. Trust in the judge's semantic assessment does not replace this output-integrity check.
4.  **Bounded Batch Processing and Consolidation:** Evidence may be screened in small batches to control context size. The screening pass must retain IDs for excerpts that contribute to a claim even when no single batch fully establishes it. A final per-claim consolidation pass judges a bounded, deduplicated set of those excerpts across batches, so complementary evidence can be combined without sending the entire 50-chunk pool to the model at once.
5.  **Labeling:** Compute the answerability score as the fraction of essential claims supported by corpus evidence. Tag questions as `Full` (all claims), `Partial` (some claims), or `None` (no claims). Keep this corpus-coverage label separate from `requires_review`: contradictions, ambiguity, malformed judge output, or unresolved temporal conflict set `requires_review: true` for audit without changing the top-level `answerability_status`.
6.  **Auditability:** Official runs must retain the method version, judge model, source index, run parameters, decomposed claims, retrieved evidence metadata/text, per-claim judgments, citations, and review reasons. In the current evaluator this means using `--include-details`. Summary-only output is suitable for smoke testing, not for the publication artifact.
7.  **Stratified Interpretation:** Use `Full`, `Partial`, and `None` as corpus-coverage strata. If a system produces a correct-looking answer for a `None` question, do not automatically count it as superior retrieval. First audit the retrieved evidence: supported evidence indicates a false-negative answerability label that must be corrected; no supporting evidence indicates parametric knowledge or an ungrounded answer, not retrieval success.

### Phase 3: Inference & Ablation Runs
We run the queries against multiple configurations to isolate the value of the multimodal components and establish strong academic baselines.

1.  **No-Context Baselines:**
    *   **Vanilla Base:** Standard LLM chatbot experience with no specialized instructions.
    *   **SOTA Base:** Strong frontier LLM answering from internal knowledge without retrieval or external tools.
2.  **BM25 Baseline (Sparse Retrieval):** Standard keyword search over the transcribed chunks to establish a traditional Information Retrieval baseline.
3.  **Vision-Only RAG:** Retrieval using only the VLM frame descriptions (isolated from the core vectors).
4.  **ASR-Only RAG:** Retrieval using only the Whisper transcripts (isolated from the core vectors).
5.  **Vector-Only RAG:** Baseline dense semantic retrieval (fusing ASR + Vision) without graph edges.
6.  **Full System (Graph-RAG):** The current hybrid system utilizing the explicit knowledge graph.

### Phase 4: Deterministic Retrieval Evaluation
Before evaluating the generated text, we evaluate the retrieval quality deterministically using the key entities (champions, items, runes) found in the Gold Answers.

*   **Entity Hit Rate (Recall):** Does the retrieved context contain the exact entities mentioned in the gold answer?
*   **Mean Reciprocal Rank (MRR):** How high up in the retrieval results were the relevant chunks?
*   **Recall@K and Precision@K:** Measure retrieval performance across the top 5, 10, and 20 chunks to provide standard Information Retrieval metrics expected by IEEE Access reviewers.

*Goal: Prove numerically that Graph-RAG finds better, more relevant evidence chunks than Vector RAG and BM25.*

### Phase 5: Generation Evaluation (LLM-as-a-Judge)
We avoid ROUGE/BLEU and instead use semantic and framework-backed grading.

1.  **Semantic Overlap:** Use **BERTScore** (`distilbert-base-uncased`) to calculate semantic similarity between the generated answer and the Gold Answer.
2.  **Reference Answer Correctness:** An automated judge (`gpt-5.4-mini`) evaluates the generated answer against the community Gold Answer on a rigid 0–2 scale (0=Incorrect, 1=Partially correct, 2=Substantially correct) to determine material correctness without being distracted by context faithfulness.
3.  **RAGAS Framework:** Use the official RAGAS framework to measure:
    *   *Faithfulness:* Does the generated answer rely *only* on facts from the retrieved video chunks?
    *   *Answer Relevance:* Does the answer directly address the user's question? 
    *(Note: To ensure robust evaluation without external rate limits, the LLM-judge for question generation uses `gpt-5.4-mini`, but the math for semantic similarity strictly utilizes a local HuggingFace `BAAI/bge-small-en-v1.5` embedding model.)*
4.  **Refusal / Fallback Rate:** Track the percentage of times the system correctly abstains from answering due to lack of evidence, and track "over-refusals" (abstaining when evidence was present).

*Engineering Note:* To support evaluating 100% of the dataset (493 questions x 7 systems), the Phase 5 evaluation scripts are highly parallelized, utilizing native HuggingFace `Dataset` async batching and `ThreadPoolExecutor` multithreading to achieve a ~100x speedup in evaluation time.

The originally planned strict-prompt, empty-context condition is excluded from the final evaluation. Its implementation returned the application's deterministic insufficient-evidence fallback before invoking the language model, so it did not constitute a valid model baseline. The preserved raw evaluation artifact retains that record for auditability. `evaluated_datasets/community_qa_dataset_final_v2.json` is an intermediate artifact containing the seven system responses but incorrect legacy answerability labels; the official consolidated publication artifact is `evaluated_datasets/community_qa_dataset_final_v3.json` after the claim/evidence answerability pass and required review resolution.

### Phase 6: Statistical Significance & Operational Efficiency
To contextualize the quality gains for publication, we perform post-processing and diagnostic analysis.

1.  **Paired Bootstrap Confidence Intervals:** For all per-question metrics (Faithfulness, Relevance, Correctness, Recall@K, etc.), we compute the mean difference between Graph-RAG and Vector-Only RAG using paired bootstrap resampling (e.g., 10,000 iterations) to report the 95% confidence interval, proving the statistical stability of the improvements.
2.  **Resource & Efficiency Diagnostics:** We report key operational metrics including mean ingestion time per video, chunk/graph size, and p50/p95 end-to-end query latency for Vector-Only vs Graph-RAG.


### Phase 7: Qualitative Positive-Control Examples
The community benchmark measures real player demand, but many real questions are broader than the currently processed video corpus. To make the corpus-coverage interpretation concrete, we add a small set of manually verified positive-control examples.

1.  **Example Selection:** Select 4-6 representative questions whose answers are clearly present in the processed videos. Each example must include:
    *   the user question,
    *   the source video,
    *   the timestamp or segment range,
    *   the retrieved evidence snippet or concise evidence summary,
    *   the system answer,
    *   a short human assessment explaining why the answer is supported.
2.  **Coverage-Control Purpose:** These examples answer a different question from the main community benchmark:
    *   Community benchmark: How well does the current corpus cover real player questions?
    *   Positive-control examples: Can the system answer correctly when the required evidence is present in the processed corpus?
3.  **Example Types:** Prefer a balanced set covering:
    *   an ASR-heavy strategic/build answer,
    *   a visually grounded gameplay-state answer,
    *   a graph/entity relation answer,
    *   a temporal/timestamp-grounded answer,
    *   one justified refusal where the evidence is absent or insufficient.
4.  **Interpretation:** This qualitative section must not be presented as statistical proof. It is a concrete demonstration that the pipeline can produce grounded answers under known evidence coverage, supporting the broader quantitative interpretation that corpus coverage is a major bottleneck.
5.  **Optional Expansion:** If reviewers, paper space, or internal review indicate that the qualitative evidence is insufficient, this positive-control set can be expanded into a 30-50 question coverage-controlled benchmark. In that expanded version, every question would be manually linked to a source segment and evaluated with the same correctness, faithfulness, refusal, and evidence-hit metrics used elsewhere.

### Phase 8: Corpus-Size Scaling Analysis
The main evaluation shows the behavior of the final 38-video knowledge base. To directly test whether limited video coverage is a practical bottleneck, we add a controlled corpus-scaling experiment.

1.  **Fixed Question Subset:** Use the questions that the full Graph-RAG system answered in the official full-corpus run:
    *   input artifact: `evaluated_datasets/community_qa_dataset_final_v3.json`;
    *   selector: `ablations.graph_rag.is_refusal == false`;
    *   expected subset size: 87 questions.
    This subset is intentionally selected from the system's actual answered set rather than the `Full`/`Partial` answerability labels. The purpose is to measure how the same answered-question slice changes when less of the video corpus is available.
2.  **Nested Corpus Conditions:** Load the existing sanitized 38-video cache and restrict the inference service to deterministic nested video subsets of size 5, 10, 20, 30, and 38. This avoids rebuilding the knowledge base while simulating progressively larger ingested corpora.
3.  **Graph-RAG Only:** Run only the full Graph-RAG system in this phase. The goal is not to compare architectures again, but to test whether the full system benefits from more ingested videos.
4.  **Output Artifact:** Store results under `knowledge_system_evaluation_v2/phase_8/` in a JSON file similar to the final dataset, but without the seven-system `ablations` object. Each question stores per-corpus-size results under `covered_videos`, including selected videos, answer, evidence sources/contexts, provisional refusal label, correctness, and faithfulness fields.
5.  **Metrics:** Compute at least:
    *   judged refusal rate,
    *   LLM-judge reference correctness on the 0-2 scale,
    *   RAGAS faithfulness over retrieved evidence contexts.
    The refusal label produced during inference is only a heuristic checkpoint. The reported refusal rate must come from the Phase 8 scoring script after the judge overwrites `is_refusal` with `refusal_method: "judge"`.
6.  **Interpretation:** Use this as supporting evidence for the paper's coverage claim. The defensible claim is that increasing the available processed video corpus should reduce abstention and tend to improve correctness on a fixed question subset. It should not be overclaimed as proof that every additional video monotonically improves every metric, because larger corpora can also introduce retrieval noise.

## 5. Handling Legacy Content in Evaluation

If a 2021 Arqade question asks about a removed item, and the system retrieves a 2021 video recommending that item, the system succeeded at its core retrieval and generation job (Multimodal RAG), even if the advice is outdated for modern gameplay.

For the purposes of the IEEE Access publication, we must strictly delineate between **Knowledge Base Currency** (having up-to-date videos) and **System Accuracy** (successfully finding and reasoning over the videos it does have).

*   **Strict Accuracy (Current Meta):** Evaluated only on questions tagged `[Current]` matching `[Current]` videos.
*   **System Faithfulness (Legacy Test):** For questions tagged `[Legacy]`, we evaluate whether the system successfully extracted the historical facts from the older videos. If the system's generated answer aligns with the legacy video but not modern gameplay, this is considered a **success** for the RAG architecture. It proves the system grounds its answers in its given corpus rather than hallucinating modern facts into historical contexts.
