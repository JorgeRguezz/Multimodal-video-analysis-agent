# Approachable Evaluation Plan for the Gameplay Knowledge System

## 1. Goal

This document defines a practical second evaluation for the multimodal gameplay knowledge system.

The evaluation should move away from internally hand-crafted or video-event-specific questions and instead use real player questions and useful community answers from external sources. The main goal is to evaluate whether the system can answer the kinds of questions that League of Legends players actually ask.

The proposed benchmark will use questions and answers from:

- MOBAFire
- Arqade / Gaming Stack Exchange

The evaluation should test:

1. Whether the system provides useful and accurate answers to real player questions.
2. Whether the system answers only when its video-derived knowledge corpus contains enough evidence.
3. Whether graph-based retrieval improves over vector-only retrieval.
4. Whether the system avoids hallucinations, outdated advice, and unsupported answers.

## 2. Core Evaluation Framing

The central evaluation question should be:

> Can a video-derived Graph-RAG knowledge system answer real League of Legends player questions accurately and usefully, while remaining grounded in its processed knowledge corpus?

This framing is stronger than the previous evaluation because:

- the questions come from external user communities,
- the reference answers are based on real community responses,
- the system is evaluated against realistic player information needs,
- answerability is separated from correctness,
- outdated or legacy game knowledge is explicitly controlled.

## 3. Data Sources

## 3.1 MOBAFire

MOBAFire is useful because it contains practical League of Legends guide content, including:

- champion builds,
- runes,
- itemization,
- matchups,
- ability usage,
- strengths and weaknesses,
- role-specific advice.

This source is especially aligned with the kind of strategic advice the system is intended to provide.

MOBAFire should be used carefully because content licensing and redistribution may be restrictive. The evaluation dataset should store source URLs and derived metadata, but should avoid publishing large copied answer bodies unless permission or license compatibility is confirmed.

## 3.2 Arqade / Gaming Stack Exchange

Arqade is useful because it provides externally sourced community questions and accepted or high-scoring answers.

Compared with MOBAFire, Arqade is likely to contain:

- more direct question-answer pairs,
- accepted answers,
- answer scores,
- question dates,
- tags,
- stable post URLs.

Arqade is valuable for reproducibility because Stack Exchange exposes structured metadata through its API.

## 4. Dataset Design

The v2 dataset should use a provenance-first schema. Every question must retain enough metadata to trace where it came from and why it was accepted.

Suggested file:

```text
knowledge_system_evaluation_v2/data/community_qa_dataset.json
```

Suggested row schema:

```json
{
  "question_id": "q_0001",
  "question": "What are some key runes and item choices for playing Aatrox effectively in the top lane?",
  "reference_answer": "...",
  "source_site": "mobafire",
  "source_url": "https://...",
  "source_question_id": "optional",
  "selected_answer_id": "optional",
  "selected_answer_score": 12,
  "selected_answer_is_accepted": true,
  "created_at": "optional",
  "last_updated_at": "optional",
  "retrieved_at": "2026-05-21",
  "tags": ["league-of-legends", "aatrox"],
  "champions": ["AATROX"],
  "topic_labels": ["runes", "items", "matchup"],
  "source_patch": "optional",
  "corpus_answerability": "full",
  "source_validity": "current",
  "notes": ""
}
```

The field `answer_type` is not necessary if the dataset construction policy is fixed:

- for Arqade, keep only accepted answers or top-scoring answers;
- for MOBAFire, keep only guide-author content or clearly high-quality/high-score content.

## 5. Dataset Construction Rules

## 5.1 Inclusion Criteria

A question-answer pair can be included if:

- it is about League of Legends;
- it asks a real gameplay, build, champion, matchup, ability, item, rune, or strategy question;
- the selected answer is concrete enough to be judged;
- the source page can be traced through a URL;
- the answer is either current, historically valid, or explicitly marked as legacy;
- the question is not purely opinion-based unless the source answer gives concrete strategic advice.

## 5.2 Exclusion Criteria

Exclude or quarantine questions if:

- the source URL is missing;
- the selected answer is unavailable;
- the answer is mostly speculation;
- the question is unrelated to League of Legends gameplay knowledge;
- the answer depends on a removed game system and cannot be labeled cleanly;
- the question is too ambiguous to evaluate;
- the content cannot be legally or ethically reused even as a derived benchmark entry.

## 6. Answerability Labels

Answerability is necessary because the system only knows what exists in its processed video corpus. A real forum question may be valid, but the system should not be expected to answer it if the processed videos do not contain enough supporting evidence.

Use this field:

```json
"corpus_answerability": "full | partial | none | unclear"
```

Definitions:

| Label | Meaning |
|---|---|
| `full` | The processed corpus contains enough evidence to answer the main question. |
| `partial` | The corpus contains some relevant evidence but misses important parts of the reference answer. |
| `none` | The corpus does not contain enough evidence to answer. |
| `unclear` | The available evidence is ambiguous or hard to judge. |

## 6.1 How To Produce Answerability Labels

The label should not be assigned by checking only the final system output. That would make the evaluation circular.

Instead:

1. Extract key entities and claims from the community reference answer.
2. Retrieve a large candidate evidence pool from the video corpus using several high-recall methods:
   - dense vector retrieval,
   - keyword/BM25 retrieval,
   - graph retrieval,
   - entity-name matching.
3. Give the question, reference answer, extracted claims, and candidate evidence to an LLM judge.
4. Ask the judge whether the corpus evidence supports the answer fully, partially, or not at all.
5. Require the judge to cite evidence chunk IDs for `full` and `partial`.
6. Human-audit all `full` labels and a sample of `partial` and `none` labels.

Suggested script:

```text
knowledge_system_evaluation_v2/scripts/label_corpus_answerability.py
```

Suggested output:

```text
knowledge_system_evaluation_v2/data/community_qa_answerability_labeled.json
```

## 7. Patch And Legacy Control

League of Legends changes frequently. Forum answers may become outdated due to:

- champion reworks,
- item removals,
- mythic item system removal,
- rune changes,
- ability changes,
- role/meta shifts,
- patch-specific advice.

Use a separate validity label:

```json
"source_validity": "current | legacy | patch_conflict | subjective | invalid"
```

Definitions:

| Label | Meaning |
|---|---|
| `current` | The answer appears valid for the current game state. |
| `legacy` | The answer was historically valid but refers to old systems or old patches. |
| `patch_conflict` | The answer conflicts with current known game facts. |
| `subjective` | The answer is advice/opinion, not an objective factual claim. |
| `invalid` | The answer is wrong, unusable, or not evaluable. |

## 7.1 How To Detect Legacy Content

Use both automatic and manual checks.

Automatic flags:

- source date older than a threshold;
- explicit old patch number;
- references to removed mythic items;
- references to removed or renamed items;
- references to removed runes;
- ability names or mechanics that no longer match current data;
- champion names that no longer match current official naming.

Manual checks:

- review high-impact questions;
- review all questions marked `patch_conflict`;
- review a random sample of `current`;
- review all cases where automatic checks disagree with LLM or human judgment.

Suggested script:

```text
knowledge_system_evaluation_v2/scripts/label_patch_validity.py
```

Main paper results should focus on `current` and possibly `subjective` questions. Legacy questions can be reported separately as a temporal-obsolescence analysis.

## 8. Baselines And Ablations

To address the main reviewer concern, the evaluation must show whether the knowledge graph adds value beyond vector retrieval.

Required systems:

| System | Purpose |
|---|---|
| Local LLM without retrieval | Measures local parametric knowledge. |
| Strong LLM without retrieval | Measures strong general League knowledge. |
| Vector-only RAG | Main non-graph retrieval baseline. |
| Graph-RAG without global graph | Tests local graph retrieval. |
| Graph-RAG without entity retrieval | Tests entity retrieval contribution. |
| Full Graph-RAG system | Main system. |

The key comparison is:

```text
Vector-only RAG vs Full Graph-RAG
```

If Full Graph-RAG improves retrieval quality, claim-level answer quality, or hallucination reduction, then the evaluation directly supports the graph contribution.

Suggested script:

```text
knowledge_system_evaluation_v2/scripts/run_community_qa_baselines.py
```

Suggested output:

```text
knowledge_system_evaluation_v2/results/community_qa_system_outputs.jsonl
```

## 9. Retrieval Metrics

Retrieval should be evaluated separately from final answer generation.

Metrics:

| Metric | Meaning |
|---|---|
| Evidence Recall@k | Whether the retrieved evidence contains support for the reference answer. |
| Evidence Precision@k | How much retrieved evidence is actually useful. |
| MRR | Rank of the first relevant evidence item. |
| nDCG@k | Ranking quality when several evidence items are relevant. |
| Entity Coverage | Whether retrieved evidence mentions required champions/items/runes. |
| Graph Relation Coverage | Whether graph retrieval recovers relevant relations. |

These metrics should be computed only for questions with:

```text
corpus_answerability = full or partial
```

For `none` questions, evaluate abstention instead.

Suggested script:

```text
knowledge_system_evaluation_v2/scripts/evaluate_community_qa_retrieval.py
```

## 10. Claim-Level Answer Evaluation

Final answers should be scored using claim-level metrics rather than only ROUGE or BERTScore.

Suggested pipeline:

1. Use an LLM to decompose the community reference answer into atomic reference claims.
2. Use an LLM to decompose the system answer into atomic system claims.
3. Use an LLM judge to classify each system claim as:
   - `supported_by_reference`,
   - `contradicted_by_reference`,
   - `not_in_reference`,
   - `unsupported_by_retrieved_evidence`.
4. Compute metrics from those labels.

Metrics:

| Metric | Meaning |
|---|---|
| Claim Precision | How many system claims are correct and supported. |
| Claim Recall | How many reference claims are recovered. |
| Claim F1 | Balanced answer quality. |
| Hallucinated Claim Rate | Claims not supported by reference or evidence. |
| Contradiction Rate | Claims that conflict with the reference. |
| Correct Abstention Rate | System refuses when corpus answerability is `none`. |
| Over-Abstention Rate | System refuses when corpus answerability is `full` or `partial`. |

Suggested scripts:

```text
knowledge_system_evaluation_v2/scripts/extract_reference_claims.py
knowledge_system_evaluation_v2/scripts/extract_system_claims.py
knowledge_system_evaluation_v2/scripts/judge_claims.py
knowledge_system_evaluation_v2/scripts/evaluate_claim_metrics.py
```

## 10.1 Actual Implementation (RAGAS & LLM Judges)

In practice, the claim-level evaluation was streamlined into a highly efficient **Phase 5** execution pipeline using established frameworks rather than purely manual scripts:

1. **RAGAS Framework:** Used to evaluate *Faithfulness* (Are claims supported by evidence?) and *Answer Relevancy* (Does it answer the question?). 
   * *Performance Note:* To bypass OpenAI embedding rate limits and achieve a 100x speedup, `Answer Relevancy` was configured to use the local open-source `BAAI/bge-small-en-v1.5` model for its underlying semantic math, while keeping `gpt-5.4-mini` for the actual intelligence.
2. **Custom Correctness Judge:** A standalone `gpt-5.4-mini` judge directly evaluates *Reference Answer Correctness* on a strict 0-2 scale.
3. **Fallback Judge:** Evaluates *Correct Abstention* vs *Over-Abstention* by strictly checking if refusals were justified by the provided context.
4. **Engineering:** All evaluation scripts were rewritten to use `ThreadPoolExecutor` and HuggingFace `Dataset` batched async execution, processing thousands of generated answers in minutes.

## 10.2 LLM-As-Judge Calibration

LLM judging is acceptable if it is controlled and calibrated.

Recommended procedure:

1. Select 50-100 questions across MOBAFire and Arqade.
2. Manually inspect reference claims, system claims, and judge labels.
3. Measure agreement between human labels and LLM labels.
4. Adjust judge prompts if needed.
5. Report that the automatic claim metrics were calibrated on a human-reviewed subset.

This avoids relying blindly on LLM-as-judge.

## 11. Human Evaluation

A full user study would require many participants. One, two, or three people are not enough to make a strong claim such as:

> Players prefer this system.

For that kind of statement, the evaluation would need a larger user group, likely at least 20-30 participants.

However, a small number of expert reviewers can still be useful for:

- auditing answerability labels;
- validating patch/legacy labels;
- calibrating LLM-as-judge outputs;
- reviewing borderline or ambiguous questions;
- checking whether forum answers are still valid.

Therefore, human evaluation should be used as supporting validation, not as the main result, unless enough participants can be recruited.

Suggested file:

```text
knowledge_system_evaluation_v2/data/human_audit_sample.json
```

Suggested script:

```text
knowledge_system_evaluation_v2/scripts/analyze_human_audit.py
```

## 12. Optional Official Factual Control Set

The main benchmark should be MOBAFire + Arqade. An official factual control set is optional.

It could contain questions based on:

- champion ability names,
- champion ability effects,
- item effects,
- rune effects,
- summoner spell effects.

What it adds:

- objective correctness checks;
- detection of basic factual hallucinations;
- protection against answers that sound useful but invent ability names or item mechanics;
- a clean control subset less affected by community opinion.

Why it is optional:

- the main paper claim is about answering real player questions;
- forum/community QA already tests usefulness;
- building and validating another dataset adds work.

Recommended compromise:

Create a small 50-100 question factual control set only if time permits. It should not replace the community QA benchmark.

## 13. Suggested Repository Structure

```text
knowledge_system_evaluation_v2/
  approachable_evaluation_plan.md
  data/
    community_qa_dataset.json
    community_qa_answerability_labeled.json
    community_qa_patch_validity_labeled.json
    human_audit_sample.json
    optional_official_factual_control.json
  scripts/
    collect_arqade_questions.py
    collect_mobafire_questions.py
    normalize_community_qa.py
    label_corpus_answerability.py
    label_patch_validity.py
    run_community_qa_baselines.py
    evaluate_community_qa_retrieval.py
    extract_reference_claims.py
    extract_system_claims.py
    judge_claims.py
    evaluate_claim_metrics.py
    analyze_human_audit.py
  results/
    system_outputs/
    retrieval_metrics/
    claim_metrics/
    audit/
  reports/
    community_qa_evaluation_summary.md
```

## 14. Minimum Viable Evaluation

If time is limited, the minimum useful version should include:

1. A provenance-preserving MOBAFire + Arqade QA dataset.
2. Deduplication and basic filtering.
3. Patch/legacy labels.
4. Corpus answerability labels.
5. Vector-only RAG vs Full Graph-RAG.
6. Claim-level LLM judge metrics.
7. Human audit of a small calibration subset.

This minimum version would already be much stronger than the first evaluation because it uses real external questions, separates missing-corpus failures from wrong answers, and directly tests the contribution of graph retrieval.

## 15. Expected Paper Results

The evaluation should produce these main tables:

### Table 1: Dataset Composition

- Number of MOBAFire questions.
- Number of Arqade questions.
- Number of unique champions.
- Question categories.
- Answerability label distribution.
- Patch validity label distribution.

### Table 2: Retrieval Performance

Rows:

- vector-only RAG,
- graph without global graph,
- graph without entity retrieval,
- full Graph-RAG.

Columns:

- Evidence Recall@5,
- Evidence Recall@10,
- MRR,
- entity coverage,
- graph relation coverage.

### Table 3: Answer Quality

Rows:

- local LLM no retrieval,
- strong LLM no retrieval,
- vector-only RAG,
- full Graph-RAG.

Columns:

- claim precision,
- claim recall,
- claim F1,
- hallucinated claim rate,
- contradiction rate,
- correct abstention,
- over-abstention.

### Table 4: Patch/Legacy Breakdown

Rows:

- current,
- legacy,
- patch conflict,
- subjective.

Columns:

- number of questions,
- system accuracy,
- hallucination rate,
- abstention behavior.

## 16. Final Evaluation Statement

The revised evaluation should be described as follows:

> We evaluate the system on real League of Legends player information needs collected from MOBAFire and Arqade. Each question retains source provenance, answer metadata, corpus answerability labels, and patch-validity labels. We compare the full Graph-RAG system against parametric and vector-only baselines using retrieval metrics, claim-level answer metrics, hallucination analysis, and abstention behavior. This design separates three factors that were conflated in the first evaluation: whether the question is answerable from the processed video corpus, whether the retrieved evidence is correct, and whether the final generated answer is faithful and useful.
