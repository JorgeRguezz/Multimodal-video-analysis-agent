# Last Additions to the Evaluation Plan

This document contains **only the additions** to make on top of `EVALUATION_PLAN 1.md` and `final_evaluation_integration 1.md`.

The existing evaluation design already covers the principal ablations, retrieval metrics, BERTScore, RAGAS faithfulness, answer relevance, temporal/legacy handling, and difficulty strata. Do not duplicate or redesign those components.

---

## 1. Add automated reference-answer correctness

### Motivation

BERTScore measures semantic similarity to the community Gold Answer, but it does not directly determine whether a generated answer is materially correct. Faithfulness answers a different question: whether an answer is supported by the retrieved video evidence.

Add a separate automated **Reference Answer Correctness** metric.

### Judge inputs

For each generated answer, provide the judge with:

- `question`
- `answer_gold`
- `generated_answer`

Do **not** provide retrieved context for this metric. The target is agreement with the accepted/community reference answer, independently of whether the system was faithful to its retrieved context.

### Output scale: 0–2

Use a three-level ordinal scale rather than 0–10. A 0–10 LLM score creates false precision: the semantic difference between, for example, 7 and 8 is difficult to define and reproduce consistently. A small rubric makes the metric more interpretable and stable.

- **0 — Incorrect or non-answer:** The response is materially wrong, contradicts the reference answer, invents important facts, or does not answer the question.
- **1 — Partially correct:** The response captures some of the relevant answer but has a material omission, ambiguity, or factual error.
- **2 — Substantially correct:** The response directly and correctly answers the question and is consistent with the Gold Answer.

Aggregate the score over all questions as a mean in `[0, 2]`. It may additionally be reported as a normalized percentage:

`normalized correctness = mean_correctness / 2 × 100`

### Required judge protocol

For reproducibility, fix and document:

- Judge model and exact version
- System prompt and rubric
- Temperature = `0`
- Output schema, including the integer score and a concise justification
- Any retry/failure policy
- The fact that answers are scored independently, not pairwise

The concise justifications should be saved in the evaluation output for auditability, but do not need to be a primary reported metric.

---

## 2. Add refusal / fallback rate

### Motivation

The RAG system may be deliberately conservative when evidence is weak. Faithfulness alone does not show whether it avoids unsupported claims by appropriately abstaining, or whether it refuses too often when useful evidence is available.

Add an automatic diagnostic based on the final answer text and/or existing inference metadata.

### Metric

For each configuration, report:

- **Overall refusal/fallback rate**
- Refusal/fallback rate by the existing difficulty strata
- Refusal/fallback rate for Current versus Legacy questions

A response counts as a refusal/fallback when it explicitly states that it cannot answer from the available evidence, lacks sufficient evidence, or emits the system’s defined low-evidence fallback behaviour.

### Interpretation

This is a diagnostic, not a replacement for faithfulness:

- A high faithfulness score plus a very high refusal rate may indicate excessive conservatism.
- A low refusal rate plus weak faithfulness may indicate unsupported answering.
- The desired behaviour depends on evidence accessibility: a higher fallback rate can be acceptable for low-accessibility questions, but is undesirable for high-accessibility questions.

---

## 3. Add paired bootstrap confidence intervals to existing metrics

### Motivation

This is **not a new quality metric** and it does not overlap with faithfulness.

Faithfulness gives a score per question:

> Is this answer supported by its retrieved evidence?

A confidence interval instead evaluates the reliability of the **difference between two systems across the full evaluation set**:

> Is the observed Graph-RAG improvement over Vector RAG likely stable, or could it result from variation in the sampled questions?

### Method

For each metric already calculated per question, compare Graph-RAG against Vector-Only RAG using paired bootstrap resampling.

For each bootstrap iteration:

1. Sample questions with replacement from the evaluation set.
2. Keep the Graph-RAG and Vector-RAG score for each sampled question paired together.
3. Compute the mean difference:

   `mean(Graph-RAG metric) − mean(Vector-RAG metric)`

4. Repeat for a sufficiently large number of iterations (e.g., 10,000).
5. Report the observed mean difference and the percentile 95% confidence interval.

### Apply to

At minimum:

- Faithfulness
- Answer relevance
- BERTScore
- Reference Answer Correctness
- Retrieval metrics where per-question values are available, such as Recall@K and MRR

### Reporting format

Example:

> Graph-RAG improved faithfulness over Vector-Only RAG by **+0.09** points (paired bootstrap 95% CI: **[+0.04, +0.14]**).

Interpretation:

- If the full interval is above zero, the result supports a stable Graph-RAG improvement.
- If the interval crosses zero, report the observed improvement but avoid claiming that it is reliably better on this dataset.

This post-processing uses the existing per-question outputs. It requires no new annotation, model, or human evaluation.

---

## 4. Add a concise efficiency and resource-cost subsection

This should be a compact practical subsection, not a full engineering benchmark.

### Ingestion and knowledge-base construction

For the processed video corpus, report:

- Mean video duration
- Mean end-to-end processing time per video
- Normalized processing factor: processing time divided by video duration
- Mean stage time per video:
  - extraction
  - pre-build sanitization
  - knowledge build
  - post-build sanitization
- Mean number of chunks per video
- Mean number of graph nodes and edges per video
- Final storage footprint of the sanitized knowledge base

### Query-time cost

Report separately for Vector-Only RAG and Full Graph-RAG:

- End-to-end query latency: p50 and p95
- Optionally, mean retrieval and generation time if already easy to log
- Failure or timeout rate, if applicable

### Purpose

The goal is to contextualize quality gains with their operational cost. Avoid broader claims about scalability unless the measurements directly support them.

---

## Scope explicitly unchanged

Do not add:

- Oracle-context evaluation
- Additional local-graph versus global-graph ablations
- Human evaluation
- Synthetic robustness/stress-test benchmarks
- Detailed manually labelled error taxonomy
- Atomic claim-level completeness metrics
- A large-scale engineering benchmark

The additions above are sufficient to strengthen the existing plan without expanding its research scope.
