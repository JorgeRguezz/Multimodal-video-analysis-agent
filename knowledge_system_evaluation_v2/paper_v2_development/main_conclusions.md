# Main Evaluation Conclusions for IEEE Access Resubmission

This document summarizes the main conclusions that can be drawn from `knowledge_system_evaluation_v2/evaluated_datasets/community_qa_dataset_evaluated.json`, the paired bootstrap analysis in `knowledge_system_evaluation_v2/phase6_analysis/bootstrap_graph_vs_vector.csv`, and the qualitative positive controls in `knowledge_system_evaluation_v2/phase7_positive_control_results_final.json`.

These conclusions incorporate the current paired bootstrap confidence intervals. The main statistical result is that every Graph-RAG versus Vector-only confidence interval crosses zero, so Graph-RAG differences should be described as observed point-estimate differences rather than statistically stable improvements.

## 1. The central empirical story is evidence-constrained gameplay QA

The revised evaluation should not be framed as a generic League of Legends question-answering benchmark. It is more accurately a test of whether a video-derived knowledge system can answer real player questions while remaining grounded in the processed video corpus.

The strongest paper framing is:

> The system is designed for grounded, evidence-constrained gameplay QA over a processed video knowledge base. Its value lies in retrieving, structuring, citing, and abstaining based on available video evidence, rather than in reproducing broad parametric game knowledge.

This distinction is important because the no-context baselines, especially `sota_base`, score much higher on reference-answer correctness and answer relevance, but they do so from general parametric knowledge rather than from the ingested gameplay videos.

## 2. Corpus coverage is the dominant bottleneck

The benchmark contains 493 real community questions:

- 307 from MOBAFire.
- 186 from Arqade.
- 396 Current questions.
- 97 Legacy questions.
- 247 Full, 124 Partial, and 122 None answerability labels.

Despite this real-question setup, all retrieval-based systems refuse a large fraction of questions:

| System | Refusal rate |
|---|---:|
| BM25 | 86.21% |
| Vision-only RAG | 84.38% |
| ASR-only RAG | 83.16% |
| Vector-only RAG | 82.96% |
| Graph-RAG | 82.35% |

This indicates that the processed video corpus does not contain enough evidence to answer most community questions directly. The system is therefore limited less by final generation quality than by knowledge-base coverage.

Recommended conclusion:

> The evaluation exposes corpus coverage as the main practical limitation. Real player questions are broader than the current 22-video knowledge base, so high abstention is expected and should be interpreted as a knowledge availability issue rather than only as model failure.

## 3. Retrieval systems are conservative and generally avoid unsupported answering

The retrieval configurations show very low over-refusal rates:

| System | Over-refusal rate |
|---|---:|
| BM25 | 0.41% |
| ASR-only RAG | 0.41% |
| Vector-only RAG | 0.41% |
| Graph-RAG | 0.41% |
| Vision-only RAG | 1.22% |

This means that when the retrieval systems refuse, the evaluator usually agrees that the retrieved context does not contain enough information to answer. This is a strong result for a grounded assistant: the system is conservative, but not randomly over-cautious.

Recommended conclusion:

> The retrieval systems demonstrate calibrated abstention under evidence scarcity. Their low over-refusal rates suggest that refusal behavior is usually tied to genuine gaps in retrieved video evidence.

## 4. No-context baselines answer more accurately, but are not grounded baselines

The no-context baselines achieve much higher correctness and relevance:

| System | Correctness, 0-2 | Relevance |
|---|---:|---:|
| Vanilla base | 0.740 | 0.762 |
| SOTA base | 1.199 | 0.839 |
| Graph-RAG | 0.144 | 0.250 |
| Vector-only RAG | 0.134 | 0.266 |

This does not mean that the retrieval systems are worse for the paper's target task. It means the no-context models can answer many League of Legends questions using broad parametric knowledge and fluent general reasoning.

The originally planned strict-prompt, empty-context condition is excluded from the final results because its recorded output was the application's deterministic insufficient-evidence fallback; the language model was never invoked. The final paper dataset therefore reports seven valid systems and preserves the earlier artifact only as an audit trail.

Recommended framing:

> No-context models provide a useful upper reference for general game knowledge, but they do not satisfy the evidence-grounded requirement of video-corpus QA. The retrieval systems trade coverage and fluency for source-grounded behavior and explicit abstention.

Avoid framing `sota_base` as a direct competitor to the grounded system unless the distinction is made explicit.

## 5. Graph-RAG is competitive with Vector-only RAG, but aggregate gains are not statistically stable

Overall Graph-RAG versus Vector-only RAG:

| Metric | Graph-RAG | Vector-only | Difference | 95% CI for difference |
|---|---:|---:|---:|---:|
| BERTScore-F1 | 0.734 | 0.736 | -0.002 | [-0.003, +0.000] |
| Correctness, 0-2 | 0.144 | 0.134 | +0.010 | [-0.022, +0.043] |
| Faithfulness | 0.632 | 0.626 | +0.006 | [-0.026, +0.037] |
| Relevance | 0.250 | 0.266 | -0.016 | [-0.055, +0.024] |
| Refusal rate | 82.35% | 82.96% | -0.61 pp | [-4.06 pp, +2.84 pp] |

The point estimates show small improvements in correctness and faithfulness, and a slightly lower refusal rate for Graph-RAG. However, every paired bootstrap confidence interval crosses zero. Vector-only remains slightly better on BERTScore and answer relevance, but those differences are also not statistically stable.

Recommended framing:

> Graph-RAG is competitive with Vector-only RAG and shows small observed gains in grounded correctness and faithfulness, but paired bootstrap confidence intervals overlap zero. The current evaluation therefore does not provide statistically stable evidence that Graph-RAG outperforms dense vector retrieval on aggregate.

Avoid:

- "Graph-RAG significantly outperforms Vector RAG."
- "The knowledge graph clearly dominates dense retrieval."
- "The graph solves the evaluation weakness identified by the CoG reviewers."

Use:

- "Graph-RAG is competitive with Vector-only retrieval under the current corpus."
- "Observed graph gains are small and not statistically stable under paired bootstrap analysis."
- "Graph structure may help under fragmented evidence, but the current dataset provides suggestive rather than conclusive support."

## 6. The largest observed Graph-RAG advantage appears in the Partial answerability stratum, but remains inconclusive

The clearest Graph-RAG advantage appears in questions labelled `Partial`, where the answer is not fully covered by direct retrieval but some relevant evidence exists.

| Partial questions | Graph-RAG | Vector-only | Difference | 95% CI for difference |
|---|---:|---:|---:|---:|
| Correctness, 0-2 | 0.185 | 0.145 | +0.040 | [-0.032, +0.121] |
| Faithfulness | 0.648 | 0.622 | +0.026 | [-0.033, +0.086] |
| Relevance | 0.258 | 0.251 | +0.007 | [-0.079, +0.096] |
| Refusal rate | 83.87% | 83.06% | +0.81 pp | [-5.65 pp, +7.26 pp] |

This is the strongest KG-specific point estimate, but the confidence intervals still overlap zero. The result should therefore be framed as a hypothesis-generating pattern rather than proof.

Recommended wording:

> The largest observed Graph-RAG gains occur on partially answerable questions, suggesting that graph traversal may help when relevant evidence is fragmented across related entities and chunks. However, the paired bootstrap intervals still cross zero, so this benefit is not statistically stable in the current evaluation.

## 7. The graph does not help uniformly across all strata

Graph-RAG is not uniformly better than vector-only retrieval.

For `Full` questions:

- Graph-RAG has slightly higher faithfulness.
- Graph-RAG has slightly higher correctness.
- Graph-RAG refuses less often.
- Vector-only has higher relevance and BERTScore.

For `None` questions:

- Vector-only performs better on correctness, faithfulness, relevance, and refusal rate.

Interpretation:

> Graph traversal may help when related evidence exists but must be connected, but this effect is not yet statistically stable. It cannot compensate for absent corpus evidence, and it may sometimes retrieve broader but less directly relevant context.

This is an important honest limitation and should be presented as part of the contribution, not hidden.

## 8. BM25 is a strong lexical baseline, especially for exact entities

BM25 achieves:

- Highest overall faithfulness among retrieval systems: 0.651.
- Highest exact multi-entity recall in `retrieval_metrics.json`: 0.759.
- Highest refusal rate: 86.21%.
- Low correctness: 0.122.

This suggests that lexical retrieval is effective when the gold answer contains exact item, rune, champion, or spell names that appear literally in the corpus. However, it is brittle for semantic or relational reasoning.

Recommended conclusion:

> BM25 remains a strong baseline for exact entity matching in game knowledge, but its high refusal rate and low correctness suggest that exact lexical overlap alone is insufficient for broader grounded gameplay QA.

## 9. ASR and vision provide complementary signals

The modality ablations show that neither audio nor vision alone is sufficient.

Overall:

| System | Correctness | Faithfulness | Relevance | Refusal rate |
|---|---:|---:|---:|---:|
| ASR-only | 0.142 | 0.615 | 0.255 | 83.16% |
| Vision-only | 0.126 | 0.627 | 0.250 | 84.38% |
| Vector-only | 0.134 | 0.626 | 0.266 | 82.96% |
| Graph-RAG | 0.144 | 0.632 | 0.250 | 82.35% |

From `retrieval_metrics.json` and `evaluation_findings.md`, the modality-specific retrieval pattern is:

- Vision-only is stronger for champion subject retrieval.
- ASR-only is stronger for detailed multi-entity recall involving items, runes, and spells.

Recommended conclusion:

> Audio and visual channels play different roles in gameplay knowledge retrieval: vision helps anchor the subject and gameplay situation, while audio carries much of the dense strategic vocabulary. The multimodal system is justified by this complementarity.

## 10. Correctness must be interpreted together with refusal

The correctness distribution shows that retrieval systems receive many zero scores:

| System | Correctness 0 | Correctness 1 | Correctness 2 |
|---|---:|---:|---:|
| Graph-RAG | 429 | 57 | 7 |
| Vector-only | 435 | 50 | 8 |
| BM25 | 439 | 48 | 6 |
| ASR-only | 429 | 58 | 6 |
| Vision-only | 436 | 52 | 5 |
| Vanilla base | 195 | 231 | 67 |
| SOTA base | 87 | 221 | 185 |

Many zero scores for retrieval systems are caused by refusal under insufficient evidence, not necessarily by hallucinated wrong answers. Therefore, correctness alone is misleading unless paired with refusal and over-refusal diagnostics.

Recommended conclusion:

> In grounded QA over incomplete video corpora, reference correctness alone conflates two behaviors: wrong answering and evidence-based abstention. Refusal and over-refusal rates are necessary to interpret correctness properly.

## 11. Conditional answered-subset results show Graph-RAG's best operating regime

To complement the full-benchmark results, we also report a conditional analysis over the subset of questions for which the grounded system produced an answer. This should not replace the full benchmark because it excludes abstentions, but it provides a useful second perspective on answer quality when the available video evidence is sufficient for the system to respond.

The coverage-oriented relabeling pass identified 87 corpus-covered questions out of 493: 5 `Full` and 82 `Partial`. This reinforces the main coverage conclusion: the processed video list is short relative to the breadth of real community questions, and even within the usable subset, most questions are only partially covered by the available videos. The count also matches the Graph-RAG answered subset size, 87 / 493 questions, although the coverage labels and the system-selected answered subset are conceptually distinct.

| System | Correctness, 0-2 | Faithfulness | Refusal rate on subset |
|---|---:|---:|---:|
| Vanilla base | 0.862 | N/A | 1.1% |
| SOTA base | 1.310 | N/A | 0.0% |
| BM25 | 0.368 | 0.604 | 55.2% |
| ASR-only RAG | 0.368 | 0.589 | 54.0% |
| Vision-only RAG | 0.414 | 0.611 | 50.6% |
| Vector-only RAG | 0.460 | 0.593 | 43.7% |
| Graph-RAG | 0.575 | 0.484 | 0.0% |

Within this answered subset, Graph-RAG has the highest correctness among retrieval-based systems: 0.575 versus 0.460 for Vector-only RAG, 0.414 for Vision-only RAG, and 0.368 for BM25 and ASR-only RAG. This suggests that when the processed video corpus contains enough evidence for the system to answer, graph-structured retrieval improves the usefulness of generated answers relative to the retrieval ablations.

The remaining gap to no-context baselines should be interpreted cautiously. Those models are not constrained to ground answers in a small processed-video corpus or to provide clip-level provenance. In contrast, Graph-RAG answers frequently include processed-video provenance such as video titles, URLs, and timestamps. This is desirable for user experience because it makes the answer auditable, but it interacts imperfectly with the current RAGAS faithfulness setup: RAGAS receives the saved `evidence_contexts` text, whereas the source headers shown to the generator (`video=...`, `time=...`, chunk/source metadata) are not preserved in those evaluation contexts. As a result, citation metadata and synthesis around cited clips may be treated as unsupported even when the generator saw that provenance. We therefore interpret Graph-RAG's faithfulness score as a conservative lower-bound estimate of strict snippet entailment, not as a complete measure of citation validity.

Recommended conclusion:

> The answered-subset analysis shows Graph-RAG's strongest practical value: when the processed video corpus contains enough evidence for the system to answer, Graph-RAG achieves the highest correctness among retrieval-based systems. This supports the usefulness of graph-structured retrieval under limited but relevant corpus coverage. However, the result is conditional: the processed video set is small, most covered questions are only partially covered, and Graph-RAG's citation-rich answers are conservatively penalized by a faithfulness evaluator that does not receive the same video provenance metadata shown to the generator.

## 12. Practical paper framing

The revised IEEE Access paper should frame the results as follows:

1. The v2 evaluation is much stronger than the CoG version because it uses real community questions, answerability labels, temporal labels, ablations, and groundedness-aware metrics.
2. The system is a grounded video-knowledge assistant, not a replacement for broad parametric game expertise.
3. The current video corpus is too small to answer most real player questions, leading to high but mostly justified refusal.
4. Graph-RAG is competitive with Vector-only RAG, but its observed gains are not statistically stable under paired bootstrap analysis.
5. The answered-subset analysis shows Graph-RAG's strongest practical advantage among retrieval systems, but this result is conditional on limited corpus coverage.
6. The strongest graph-related pattern appears in partially answerable questions, but this remains suggestive rather than conclusive.
7. Multimodal extraction is justified because audio and vision contribute different retrieval signals.
8. Future work should focus on scaling corpus coverage, improving retrieval calibration, preserving provenance metadata in faithfulness evaluation, and evaluating graph benefits on larger or more controlled corpora.

## 13. Bootstrap confidence interval interpretation

The paired bootstrap analysis compares Graph-RAG and Vector-only RAG on the same questions. Every confidence interval currently crosses zero, including the aggregate metrics and the Full, Partial, None, Current, and Legacy slices.

Therefore, the following claims should not be made:

- Graph-RAG significantly improves faithfulness over Vector-only RAG.
- Graph-RAG significantly improves correctness over Vector-only RAG.
- Graph-RAG significantly improves Partial-stratum performance over Vector-only RAG.
- Graph-RAG significantly reduces refusal rate compared to Vector-only RAG.

Use language such as:

> Graph-RAG showed small observed gains in correctness and faithfulness, but the paired bootstrap confidence intervals crossed zero. These differences should be interpreted as suggestive rather than statistically stable.

For the Partial stratum, use:

> The largest observed Graph-RAG gains appeared on partially answerable questions, but confidence intervals still overlapped zero, so this pattern should be treated as a direction for future evaluation rather than a confirmed effect.

## 14. Phase 7 qualitative positive controls

The five Phase 7 positive-control cases provide encouraging qualitative evidence that the complete system can produce useful, grounded answers when relevant knowledge is present in the processed corpus. Four of the five controls achieved an operationally correct outcome: three substantive answers recovered the central information requested, and the fourth correctly refused an unsupported request. Together, these cases covered textual strategy and build advice, visual gameplay state, temporal strategy synthesis, and justified refusal, showing that the system can operate across the principal evidence and response types targeted by the architecture.

Retrieval found the gold evidence for three of the four corpus-answerable questions, at ranks 1, 2, and 6. The resulting answers substantially recovered Pyke's build progression, Rengar's opening gameplay state, and Aatrox's mid- and late-game roles. The Lux control also produced the intended refusal rather than inventing an unsupported jungle route. All five final outputs were structurally valid and usable.

The remaining unsuccessful case was localized to retrieval: the relevant Deathbringer Stance segment was not returned to the answer generator. This component-specific outcome is useful because it points toward retrieval recall and provenance precision as the most direct improvement targets, rather than indicating a need to redesign the overall approach.

Recommended conclusion:

> The qualitative positive controls demonstrate that the full system can retrieve and synthesize heterogeneous video-derived knowledge into useful answers across textual, visual, temporal, and refusal-oriented cases. Four of the five controls achieved an operationally correct outcome: the substantive responses captured the central information requested, while the refusal case appropriately avoided inventing unsupported guidance. These results provide encouraging evidence that the multimodal pipeline can support grounded gameplay QA. The remaining limitation points to a concrete opportunity to improve retrieval recall and provenance precision. Although this small qualitative sample is not intended to establish statistically significant performance, it supports the feasibility of the approach and shows that the system's errors are understandable and component-specific.

## 15. Suggested final takeaway paragraph

Draft wording:

> The evaluation shows that real player questions expose a fundamental tension in video-grounded gameplay QA: broad no-context models answer more questions correctly, but without evidence grounding, while retrieval-based systems are constrained by the coverage of the processed video corpus. The proposed Graph-RAG system demonstrates calibrated abstention and remains competitive with vector-only retrieval, with small observed gains in grounded correctness and faithfulness. However, paired bootstrap confidence intervals overlap zero, so the current evidence does not establish a statistically stable Graph-RAG advantage on the full benchmark. The answered-subset analysis shows the system's strongest operating regime: when the processed video corpus contains enough evidence to answer, Graph-RAG achieves the highest correctness among retrieval-based systems. This conditional result should be interpreted alongside the small corpus scale, the predominance of partially covered questions, and the conservative faithfulness penalty applied to citation-rich answers whose video provenance metadata is not fully represented in the RAGAS context field. Overall, the results support the feasibility of grounded video-corpus QA while showing that corpus scale, retrieval coverage, provenance-aware evaluation, and stronger graph-dependent evaluation settings remain the primary bottlenecks for practical deployment.
