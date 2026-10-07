# Proposed Visualizations for IEEE Access Evaluation

This document proposes figures and tables for explaining the evaluation results from `knowledge_system_evaluation_v2/community_qa_dataset_evaluated.json`.

The main goal is to make the paper's empirical story easy to understand:

- The benchmark uses real community questions.
- The video corpus has limited coverage.
- Retrieval systems are conservative and grounded.
- Graph-RAG is competitive with Vector-only RAG, but paired bootstrap CIs show no statistically stable advantage in the current evaluation.
- Audio, vision, vector retrieval, lexical retrieval, and graph retrieval contribute different signals.

## Figure 1. Evaluation Dataset Composition

Type: grouped bar chart or three compact horizontal bar charts.

Panels:

1. Source distribution:
   - MOBAFire: 307.
   - Arqade: 186.
2. Answerability distribution:
   - Full: 247.
   - Partial: 124.
   - None: 122.
3. Temporal distribution:
   - Current: 396.
   - Legacy: 97.

Purpose:

This figure establishes that the evaluation is not an internally hand-crafted QA set. It uses real community questions, then controls for corpus answerability and game-version temporal effects.

Suggested caption:

> Composition of the 493-question community benchmark. Questions are drawn from MOBAFire and Arqade, labelled by corpus answerability and temporal status to separate retrieval coverage from game-version currency.

## Figure 2. Main Metric Heatmap Across Systems

Type: heatmap.

Rows:

- Vanilla base.
- SOTA base.
- BM25.
- ASR-only RAG.
- Vision-only RAG.
- Vector-only RAG.
- Graph-RAG.

Columns:

- BERTScore-F1.
- Correctness, 0-2.
- Faithfulness.
- Answer relevance.
- Refusal rate.
- Over-refusal rate.

Notes:

- Use a diverging or sequential colormap, but make refusal rate visually distinct because lower is not always better.
- Gray out faithfulness for no-context baselines because faithfulness to retrieved evidence is not applicable.
- Consider annotating each cell with rounded values.

Purpose:

This is the compact overview table/figure. It shows the key contrast: no-context models are strong on reference correctness/relevance, while retrieval systems are evidence-constrained and highly conservative.

Suggested caption:

> Aggregate performance across all 493 questions. No-context baselines achieve higher reference-answer correctness, while retrieval-based systems exhibit high abstention and measurable faithfulness to retrieved video evidence.

## Figure 3. Correctness Versus Refusal Trade-off

Type: scatter plot.

X-axis:

- Refusal rate.

Y-axis:

- Mean correctness score, 0-2.

Points:

- One point per system.

Color:

- No-context baselines.
- Retrieval-based systems.

Optional encoding:

- Point size = answer relevance.
- Point label = system name.

Purpose:

This plot makes the central trade-off visible. No-context systems answer almost everything and achieve high correctness. Retrieval systems refuse often because they require evidence. This prevents reviewers from reading low correctness as simple system incompetence.

Suggested caption:

> Relationship between answer coverage and reference-answer correctness. Retrieval systems operate in a high-refusal regime because they are constrained to answer from retrieved video evidence, whereas no-context baselines answer broadly from parametric knowledge.

## Figure 4. Graph-RAG Versus Vector-only by Answerability Stratum

Type: grouped bar chart or small-multiple delta plot.

X-axis:

- Full.
- Partial.
- None.

Y-axis options:

- Correctness, 0-2.
- Faithfulness.
- Answer relevance.
- Refusal rate.

Recommended layout:

- Four small panels, one metric per panel.
- Each panel compares Graph-RAG and Vector-only.

Key values:

Partial questions:

- Correctness: Graph-RAG 0.185, Vector-only 0.145.
- Faithfulness: Graph-RAG 0.648, Vector-only 0.622.
- Relevance: Graph-RAG 0.258, Vector-only 0.251.
- Refusal: Graph-RAG 83.87%, Vector-only 83.06%.

Purpose:

This is still an important KG-focused figure, but it should be presented as an observed pattern rather than proof. It shows that the largest Graph-RAG point-estimate advantage appears for Partial questions, where evidence exists but is incomplete or distributed.

Suggested caption:

> Graph-RAG and Vector-only RAG performance stratified by corpus answerability. The largest observed Graph-RAG advantage appears on partially answerable questions, suggesting a possible role for graph traversal under fragmented evidence; paired bootstrap intervals should be reported to show that this effect is not statistically stable in the current dataset.

## Figure 5. Graph-RAG Minus Vector-only Delta Plot

Type: horizontal bar chart with confidence intervals.

Metrics:

- BERTScore-F1.
- Correctness.
- Faithfulness.
- Answer relevance.
- Refusal rate.

Y-axis:

- Metric.

X-axis:

- Mean difference: Graph-RAG minus Vector-only.

Add:

- 95% paired bootstrap confidence intervals.
- Vertical zero line.

Purpose:

This is the main statistical comparison. It shows that the Graph-RAG versus Vector-only point-estimate differences are small and that every headline confidence interval crosses zero.

Current point estimates:

| Metric | Graph-RAG minus Vector-only | 95% CI |
|---|---:|---:|
| BERTScore-F1 | -0.002 | [-0.003, +0.000] |
| Correctness | +0.010 | [-0.022, +0.043] |
| Faithfulness | +0.006 | [-0.026, +0.037] |
| Relevance | -0.016 | [-0.055, +0.024] |
| Refusal rate | -0.006 | [-0.041, +0.028] |

Suggested caption:

> Paired differences between Graph-RAG and Vector-only RAG across the full benchmark. Error bars show paired bootstrap 95% confidence intervals.

Note:

This figure should make the non-stable result visually clear: every interval crosses the zero line. That is a useful and honest result, not a failure of the figure.

## Figure 6. Correctness Score Distribution

Type: stacked bar chart.

Rows or X-axis:

- Each ablation.

Stack segments:

- Correctness score 0.
- Correctness score 1.
- Correctness score 2.

Key counts:

| System | Score 0 | Score 1 | Score 2 |
|---|---:|---:|---:|
| Graph-RAG | 429 | 57 | 7 |
| Vector-only | 435 | 50 | 8 |
| BM25 | 439 | 48 | 6 |
| ASR-only | 429 | 58 | 6 |
| Vision-only | 436 | 52 | 5 |
| Vanilla base | 195 | 231 | 67 |
| SOTA base | 87 | 221 | 185 |

Purpose:

This shows that retrieval systems receive many zero scores, while no-context systems produce more partial/substantial answers. It should be discussed together with refusal rates to avoid misleading interpretation.

Suggested caption:

> Distribution of LLM-judged reference-answer correctness. Retrieval-based systems receive many zero scores largely because they abstain when retrieved evidence is insufficient.

## Figure 7. Refusal and Over-refusal by System

Type: paired bar chart.

For each system:

- Refusal rate.
- Over-refusal rate.

Alternative:

- Use a main bar for refusal rate and a small inset for over-refusal because over-refusal values are very small.

Purpose:

This is the clearest figure for calibrated abstention. It shows that refusal is high, but over-refusal is low.

Suggested caption:

> Refusal and over-refusal rates. Retrieval-based systems frequently abstain under limited evidence, but over-refusal remains low, indicating that most abstentions correspond to genuine evidence gaps.

## Figure 8. Modality Ablation Comparison

Type: grouped bar chart or radar chart.

Systems:

- ASR-only.
- Vision-only.
- Vector-only.
- Graph-RAG.

Metrics:

- Correctness.
- Faithfulness.
- Relevance.
- Refusal rate.

Optional second panel:

- Entity recall from `retrieval_metrics.json`:
  - BM25: 0.759.
  - Vector-only: 0.743.
  - Graph-RAG: 0.736.
  - ASR-only: 0.672.
  - Vision-only: 0.632.

Purpose:

This figure supports the multimodal design argument. It shows that neither ASR nor vision alone is sufficient and that multimodal retrieval changes behavior.

Suggested caption:

> Modality ablation results. Audio and visual evidence provide complementary retrieval signals, with ASR carrying dense strategic vocabulary and vision anchoring gameplay context.

## Figure 9. Graph-RAG Evidence Source Composition

Type: stacked bar chart or donut chart.

Source counts from Graph-RAG evidence:

- Entity graph: 6371.
- Dense chunk: 5359.
- Global graph: 2760.
- Visual support: 947.

Purpose:

This explains what Graph-RAG actually does operationally. It shows that Graph-RAG is not only dense retrieval with a graph label; multiple retrieval branches contribute evidence.

Suggested caption:

> Evidence-source composition for Graph-RAG. Retrieved contexts combine entity-graph expansion, dense chunk retrieval, global graph retrieval, and visual-support retrieval.

Recommendation:

Prefer a stacked bar chart over a donut chart if IEEE formatting space is tight.

## Figure 10. Current Versus Legacy Performance

Type: small multiples or grouped bars.

Panels:

- Correctness.
- Faithfulness.
- Refusal rate.

Systems:

- BM25.
- Vector-only.
- Graph-RAG.
- Vanilla base.
- SOTA base.

Purpose:

This figure explains the temporal/currency dimension. It can show that legacy questions are harder for retrieval systems and that parametric models may answer from broad historical knowledge rather than the processed video corpus.

Suggested caption:

> Performance split by temporal status. Current and Legacy questions are evaluated separately to distinguish system grounding from the currency of the available video corpus.

## Figure 11. Answerability Flow Diagram

Type: pipeline-style Sankey or flow diagram.

Flow:

1. 493 total questions.
2. Split by source: MOBAFire and Arqade.
3. Split by temporal status: Current and Legacy.
4. Split by answerability: Full, Partial, None.
5. Optional final split: answered/refused by Graph-RAG.

Purpose:

This gives readers an intuitive picture of how many real questions survive each evaluation condition.

Suggested caption:

> Evaluation set stratification from raw community questions to corpus-answerability groups and Graph-RAG answer/refusal outcomes.

## Table 1. Main Evaluation Results

Type: paper table.

Rows:

- BM25.
- ASR-only.
- Vision-only.
- Vector-only.
- Graph-RAG.
- Vanilla base.
- SOTA base.

Columns:

- BERTScore-F1.
- Correctness, 0-2.
- Faithfulness.
- Relevance.
- Refusal rate.
- Over-refusal rate.

Purpose:

This is the formal numeric table corresponding to the heatmap. Use bold sparingly:

- Bold best retrieval-based score separately from best no-context score.
- Do not bold no-context and retrieval systems in one pool without explanation, because they measure different operating regimes.

## Table 2. Answerability-Stratified Graph-RAG vs Vector-only

Type: paper table.

Rows:

- Full.
- Partial.
- None.

Columns:

- Correctness Graph-RAG.
- Correctness Vector-only.
- Faithfulness Graph-RAG.
- Faithfulness Vector-only.
- Refusal Graph-RAG.
- Refusal Vector-only.

Purpose:

This table supports a nuanced KG discussion. It shows where Graph-RAG has the largest observed point-estimate gains, while the accompanying CI plot should make clear that those gains are not statistically stable in the current dataset.

## Table 3. Representative Qualitative Examples

Type: qualitative examples table.

Rows:

1. A case where Graph-RAG is stronger than Vector-only.
2. A case where Vector-only is stronger than Graph-RAG.
3. Correct refusal due to missing evidence.
4. Legacy/currency example.
5. Modality example where ASR or vision retrieves distinct useful context.

Columns:

- Question.
- Answerability label.
- Gold answer summary.
- Graph-RAG answer summary.
- Vector-only answer summary.
- Retrieved evidence summary.
- Interpretation.

Purpose:

Reviewers explicitly asked for concrete examples. This table addresses that criticism more directly than plots.

## Recommended Figure Set for the Paper

If space is limited, prioritize:

1. Dataset composition.
2. Main metrics heatmap or main results table.
3. Correctness versus refusal scatter.
4. Graph-RAG vs Vector-only by answerability stratum.
5. Graph-RAG minus Vector-only with bootstrap CIs.
6. Graph-RAG evidence source composition.
7. Qualitative examples table.

If space allows, add:

8. Modality ablation comparison.
9. Current versus Legacy split.
10. Correctness distribution stacked bars.

## Visual Style Recommendations

- Use consistent system ordering across all figures:
  1. Vanilla base.
  2. SOTA base.
  3. BM25.
  4. ASR-only.
  5. Vision-only.
  6. Vector-only.
  7. Graph-RAG.
- Use one color family for no-context baselines and another for retrieval systems.
- Highlight Graph-RAG with a consistent accent color.
- Report percentages as percentages, not decimals, for refusal and over-refusal.
- Report correctness as `0-2`, or normalize it to percentage only if clearly stated.
- For IEEE readability, avoid overly dense legends and use direct labels where possible.

## Cautions

- Do not plot faithfulness for no-context systems as zero. It is not applicable.
- Do not use BERTScore as the primary success metric; it barely separates systems and does not capture grounding.
- Do not interpret high refusal as purely bad. In this task, refusal must be read with over-refusal and answerability labels.
- Do not claim Graph-RAG superiority from point estimates. The paired bootstrap intervals currently cross zero for all headline Graph-RAG versus Vector-only comparisons.
- Do not mix no-context and retrieval baselines without explaining that they optimize different objectives.

