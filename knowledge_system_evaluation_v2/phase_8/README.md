# Phase 8 — Graph-RAG corpus-size scaling evaluation

This phase tests whether Graph-RAG performance changes as the number of available
processed videos increases.

The experiment does **not** rebuild the knowledge base. It loads the existing
sanitized 38-video cache and filters the inference service's loaded video stores
at runtime.

## Subset

Input dataset:

```text
knowledge_system_evaluation_v2/evaluated_datasets/community_qa_dataset_final_v3.json
```

Selector:

```python
record["ablations"]["graph_rag"]["is_refusal"] is False
```

This selects the 87 questions that Graph-RAG answered in the full-corpus run.

## Corpus sizes

The default corpus sizes are:

```text
5, 10, 20, 30, 38
```

The video order is deterministic for a given seed. The default seed is `0`.

## Outputs

Inference output:

```text
knowledge_system_evaluation_v2/phase_8/graph_rag_corpus_scaling_results.json
```

Scored output:

```text
knowledge_system_evaluation_v2/phase_8/graph_rag_corpus_scaling_results_scored.json
```

Summary outputs:

```text
knowledge_system_evaluation_v2/phase_8/graph_rag_corpus_scaling_summary.json
knowledge_system_evaluation_v2/phase_8/graph_rag_corpus_scaling_summary.csv
```

The new result JSON uses `covered_videos` instead of `ablations`.


## Optional safety test — small Graph-RAG inference run

Before the full inference run, you can validate the scaling path with 2 questions
and corpus sizes 5 and 38. This writes to a separate test file by default.

```bash
python knowledge_system_evaluation_v2/phase_8/run_graph_rag_corpus_scaling.py \
  --input knowledge_system_evaluation_v2/evaluated_datasets/community_qa_dataset_final_v3.json \
  --test \
  --resume
```

Default test output:

```text
knowledge_system_evaluation_v2/phase_8/graph_rag_corpus_scaling_results_test.json
```

## Step 1 — Run Graph-RAG inference

```bash
python knowledge_system_evaluation_v2/phase_8/run_graph_rag_corpus_scaling.py \
  --input knowledge_system_evaluation_v2/evaluated_datasets/community_qa_dataset_final_v3.json \
  --output knowledge_system_evaluation_v2/phase_8/graph_rag_corpus_scaling_results.json \
  --sizes 5 10 20 30 38 \
  --seed 0 \
  --resume
```

This runs:

```text
87 questions × 5 corpus sizes = 435 Graph-RAG generations
```

## Step 2 — Score refusal, correctness, and faithfulness

```bash
python knowledge_system_evaluation_v2/phase_8/evaluate_graph_rag_corpus_scaling_metrics.py \
  --input knowledge_system_evaluation_v2/phase_8/graph_rag_corpus_scaling_results.json \
  --output knowledge_system_evaluation_v2/phase_8/graph_rag_corpus_scaling_results_scored.json \
  --metrics refusal correctness faithfulness \
  --resume
```

This uses the same correctness rubric as the main evaluation and RAGAS
Faithfulness over the retrieved evidence contexts.

## Step 3 — Summarize metrics by corpus size

```bash
python knowledge_system_evaluation_v2/phase_8/summarize_graph_rag_corpus_scaling.py \
  --input knowledge_system_evaluation_v2/phase_8/graph_rag_corpus_scaling_results_scored.json \
  --output knowledge_system_evaluation_v2/phase_8/graph_rag_corpus_scaling_summary.json \
  --csv-output knowledge_system_evaluation_v2/phase_8/graph_rag_corpus_scaling_summary.csv
```
