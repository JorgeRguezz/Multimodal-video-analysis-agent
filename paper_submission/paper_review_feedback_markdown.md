# Paper Review Feedback

**Submission:** 182  
**Title:** *From Gameplay to Knowledge Graph: Multimodal Retrieval-Augmented QA for League of Legends*

---

## Review Scores Overview

| Review | Score | Recommendation | Best Paper Nomination |
|---|---:|---|---|
| Review 1 | -1 | Weak reject | No |
| Review 2 | 0 | Borderline paper | No |
| Review 3 | 1 | Weak accept | No |
| Review 4 | -1 | Weak reject / Metareview | No |

---

## High-Level Takeaways

The reviewers generally agree that the **motivation is strong** and that the proposed pipeline is sensible for the problem of extracting useful knowledge from gameplay videos. They see the topic as relevant to the Conference on Games and recognize the potential of using multimodal extraction, knowledge graphs, and RAG for gameplay question answering.

However, the main concerns are:

1. **The evaluation does not convincingly prove that the knowledge graph adds value.**
2. **The graph construction process is not described in enough detail.**
3. **The paper lacks qualitative examples** such as sample queries, answers, chunks, local graphs, global graphs, or extracted relationships.
4. **The sampling strategy may be too sparse** to capture important gameplay details.
5. **The code/demo was not made anonymously available**, despite blind-review-compatible options.
6. **The references rely heavily on arXiv/preprints and web sources**, which should be reduced where possible.

---

# Review 1

## Overall Evaluation

**Score:** -1  
**Decision:** Weak reject

## Summary

The authors present an approach for extracting knowledge from gameplay videos into a knowledge graph and using this graph as RAG context for LLMs. The system is demonstrated on League of Legends videos.

## Strong Points

- The authors are trying to move LLMs beyond their “vanilla” gameplay understanding using a relatively lightweight approach.

## Weak Points

### Sampling Strategy

The input videos are split into 30-second segments with 5 frames, one every six seconds. The reviewer considers this a very rough sampling strategy that may miss a lot of important gameplay detail.

### Knowledge Graph Quality

The reviewer questions whether the generated knowledge graph is actually useful.

According to Table I, the proposed approach performs best only on one metric: **faithfulness**, which is used for detecting hallucinations. On other metrics, especially **relevancy**, the proposed system performs much worse.

The reviewer’s main concern is that the quality of the knowledge graph depends heavily on the quality of the data extraction process performed by the AI models. Recent work on games and VLMs suggests that VLMs are not yet very strong at interpreting game visuals, such as the work of Taesiri et al. on VideoGameBunny.

As a result, the reviewer argues that the current evaluation may be measuring:

- The quality of the underlying models.
- The overhead of converting extracted information into a knowledge graph.

Rather than clearly proving:

- That a knowledge graph improves LLM performance.

The reviewer considers the idea promising but finds the current data unconvincing.

## Novelty

Automatically extracting a knowledge graph that can be used as RAG context for LLMs.

## Technical Soundness

The approach appears to be technically sound.

## Potential Impact

The potential impact is hard to judge at the moment because the current performance is not strong.

The reviewer recommends larger-scale experiments, possibly using:

- Better LLMs for data extraction.
- A manually crafted knowledge graph to demonstrate the benefit of the KG structure.

The reviewer argues that the paper is trying to solve two problems at once:

1. Automatically extracting the knowledge graph.
2. Using a knowledge graph to improve LLM performance.

They recommend first demonstrating the value of the second point, and only then focusing on automatic KG extraction.

## Minor Comments

- Some text in Figures 2–4 is too small to read, even after zooming to 300%.

## Best Paper Nomination

**Selection:** No

---

# Review 2

## Overall Evaluation

**Score:** 0  
**Decision:** Borderline paper

## Main Feedback

The reviewer frames the problem as highly relevant to the video game industry. Complex gameplay mechanics often require players to rely on external knowledge sources such as Twitch streams, YouTube tutorials, and community forums. However, extracting useful knowledge from these sources is still mostly manual.

This is especially true for MOBA games such as League of Legends and Dota 2, where frequent patch updates cause the META to evolve quickly and make previous strategic guides obsolete.

The reviewer notes that structured game replays are often unavailable, so systems that answer user questions need to combine several data sources, including video, documents, and audio.

They agree that RAG is a general solution that can help, but that video games introduce a multimodal challenge because information is spread across videos, documents, audio, and visual gameplay elements.

## Positive Comments

The reviewer considers the paper’s problem and proposed direction suitable for the Conference on Games.

They describe the paper as proposing a generic video summarization system based on a **Multimodal Graph-RAG architecture** for advanced, personalized knowledge retrieval.

They also state that:

- The article is easy to read.
- The ideas and motivations are great.
- The proposed pipeline makes sense.
- The contribution is technical and based on several existing building blocks.
- The system is tested on 22 videos and compared against general models such as RAG-Anything using a manually built ground truth and the RAGAS evaluation framework.

## Drawbacks

The reviewer identifies several important weaknesses:

- The methodology is described too generally and does not go into enough detail.
- There is not a single concrete example, such as:
  - A query.
  - An answer.
  - A local graph.
  - A global graph.
  - A chunk.
  - A summary.
- There is no qualitative evaluation showing actual questions and answers.
- No data are available.
- The GitHub link is anonymized as `XXX`, but the code or a demo website could have been shared anonymously.
- Graph construction and merging into the global graph need more detail.
- The figures are useful and visually nice, especially Figures 1, 2, and 3, but they would be stronger if accompanied by examples.

## Suggested References

The reviewer suggests the following related works:

### Watch me playing, I am a professional: a first study on video game live streaming

**Authors:** M. Kaytoue, A. Silva, L. Cerf, W. Meira Jr, C. Raïssi  
**Venue:** Proceedings of the 21st International Conference on World Wide Web  
**Pages:** 1181–1188

Reviewer note: This is described as the first paper on Twitch showing that users also come to the platform to learn from experts.

### What did I do wrong in my MOBA game? Mining patterns discriminating deviant behaviours

**Authors:** Cavadenti et al.  
**Venue:** DSAA 2016

Reviewer note: This work extracts correlations from heroes/items and movement patterns in Dota 2.

### Exceptional contextual subgraph mining

**Authors:** Kaytoue et al.  
**Venue:** Machine Learning, 106(8), 1171–1211

Reviewer note: This work extracts knowledge as movement patterns and their context.

### Team Dynamics in DotA2 Through Attention Mechanism

**Authors:** Mortellier et al.  
**Year:** 2024

**Link:**  
https://link.springer.com/chapter/10.1007/978-3-031-86692-0_9

### New York Times League of Legends Interactive Article

**Link:**  
https://www.nytimes.com/interactive/2014/10/10/technology/league-of-legends-graphic.html

## Best Paper Nomination

**Selection:** No

---

# Review 3

## Overall Evaluation

**Score:** 1  
**Decision:** Weak accept

## Summary

The paper presents a pipeline for multimodal extraction of gameplay content in order to build a knowledge graph used for question answering.

The reviewer sees potential in the proposed method because it could help video game players interactively engage with gameplay content and find the information they need without having to watch long gameplay videos.

## Points That Can Be Improved

### Knowledge Graph Construction

Automatic construction of the knowledge graph is an important aspect of the paper, but some key steps appear to be overlooked in the description.

For example, merging multiple local graphs into one global graph is an important step in knowledge graph construction. The reviewer suggests that the merging approach should be discussed, even briefly, or that the specific merging method should be mentioned.

The same applies to the sanitization step.

The reviewer asks whether any of the following were involved:

- Named entity recognition.
- Entity linking.
- Other entity normalization or linking processes.

### Definition of Extracted Information

The reviewer suggests adding a clearer definition of the expected extracted information.

The paper mentions that the graph is initially built from chunk text, which resembles open information extraction. However, additional steps are performed before the final knowledge graph, including sanitization.

The reviewer says that, even without defining a full ontology, the authors should be more precise about what type of information is expected in the resulting knowledge graph.

This could include:

- Relevant node types.
- Relevant edge types.
- Gameplay-specific entities.
- Event-related information.
- Information needed for downstream question answering.

### Event and Time-Bounded Information

The evaluation section discusses many event-focused queries. However, the reviewer finds it difficult to identify how events or time-bounded information are extracted and stored in the knowledge graph construction method.

This should be clarified.

### Contribution Framing

The reviewer suggests that the first two contribution points could be merged, because they may sound redundant.

Both contributions relate to the broader idea of proposing a method for automatic knowledge graph construction from multimodal data sources.

### Reproducibility

The method is described in a way that makes reproducibility feasible, but not precise.

Some steps are vaguely described, especially in the evaluation protocol.

The reviewer says it would help to include sample questions from the curated question-answer evaluation.

### Citation Style

The reviewer comments that citations do not need to be accumulated at the end of every sentence, especially in the related work section.

This style is usually discouraged unless all references support the same claim.

### Use of Online Articles and Preprints

For a peer-reviewed paper, the reviewer recommends avoiding online articles as references unless absolutely necessary, because web articles may be biased or uncontrolled.

The same concern applies, to some extent, to preprints.

The reviewer notes that while citing preprints may sometimes be unavoidable, it is good practice to:

- Cite peer-reviewed versions when available.
- Find alternative peer-reviewed related work.
- Keep preprints to a minimum.

The reviewer notes that more than half of the paper’s citations are arXiv preprints.

### Anonymous Repository

The reviewer suggests using an anonymized repository to share resources under blind review.

Suggested platform:

https://anonymous.4open.science/

## Best Paper Nomination

**Selection:** No

---

# Review 4

## Overall Evaluation

**Score:** -1  
**Decision:** Weak reject / Metareview

## Metareview

All three reviewers find the motivation compelling and the pipeline concept sensible.

They agree that the problem is real:

- MOBA game knowledge becomes obsolete rapidly.
- Replay files are often unavailable.
- Manually watching hours of video is inefficient.

The end-to-end architecture is considered a reasonable response to this gap.

## Central Empirical Concern

The central empirical concern is that the evaluation results do not convincingly demonstrate that the knowledge graph adds value.

The proposed approach outperforms baselines on only one reported metric:

- Faithfulness.

However, it performs worse on other metrics, especially:

- Relevancy.

Because VLMs are known to struggle with game visual interpretation, the evaluation may mainly reflect the quality of the underlying models rather than the value of the knowledge graph itself.

## Methodological Transparency

Another shared concern is methodological transparency.

The graph construction process is unclear, especially:

- Local-to-global graph merging.
- The sanitization step.

The paper does not provide examples of:

- Queries.
- Answers.
- Graph structures.
- Extracted chunks.

This makes it difficult to assess what the system actually produces.

## Code Availability

The repository link is anonymized as `XXX`.

The reviewers note that anonymous sharing through platforms such as `anonymous.4open.science` is possible and expected under blind review.

## References

The metareview acknowledges that many citations are arXiv preprints because the field is moving quickly.

However, this still raises issues. Ideally, preprints should be replaced with peer-reviewed versions when available.

## Recommendation for Resubmission

The metareview encourages a resubmission that:

1. Decouples the two research problems:
   - Automatically extracting the knowledge graph.
   - Showing that a knowledge graph improves LLM performance.
2. Provides qualitative examples.
3. Strengthens the evaluation with a manually constructed knowledge graph baseline.

## Best Paper Nomination

**Selection:** No

---

# Consolidated Revision Priorities

## Priority 1: Prove That the Knowledge Graph Adds Value

The biggest issue is that the paper does not convincingly isolate the benefit of the KG.

Recommended additions:

- Add a manually constructed KG baseline.
- Compare:
  - No RAG.
  - Vector-only RAG.
  - Automatically extracted KG-RAG.
  - Manually constructed KG-RAG.
- Add ablation studies showing the impact of:
  - Graph traversal.
  - Entity linking.
  - Temporal filtering.
  - Chunk retrieval.
  - Sanitization.

## Priority 2: Add Concrete Examples

The reviewers repeatedly ask for examples.

Recommended additions:

- One complete worked example showing:
  - Video segment.
  - Sampled frames.
  - ASR transcript.
  - Segment summary.
  - Extracted chunk.
  - Extracted entities.
  - Extracted relationships.
  - Local graph.
  - Global graph merge.
  - User query.
  - Retrieved evidence.
  - Final answer.
- A qualitative evaluation table with several representative questions and answers.

## Priority 3: Explain Graph Construction in More Detail

The graph pipeline needs to be more transparent.

Recommended additions:

- Define expected node types.
- Define expected edge types.
- Explain entity normalization.
- Explain entity linking.
- Explain local-to-global graph merging.
- Explain sanitization.
- Clarify how events and time-bounded information are stored.

## Priority 4: Improve the Evaluation

Recommended additions:

- More questions in the QA benchmark.
- Clear description of how questions were created.
- Example reference answers.
- Breakdown by query type:
  - Event-focused.
  - Entity-focused.
  - Relationship-focused.
  - Temporal queries.
- Larger video dataset if possible.
- Human evaluation of answer quality and grounding.
- Evidence localization accuracy, such as timestamp correctness.

## Priority 5: Improve Reproducibility

Recommended additions:

- Anonymous code repository.
- Anonymous demo.
- Example dataset or sample processed videos.
- Released evaluation questions and reference answers.
- Configuration details for each model and pipeline stage.

## Priority 6: Improve Figures and Writing

Recommended edits:

- Make text in Figures 2–4 larger.
- Add figure callouts with a concrete example.
- Reduce citation clustering.
- Replace arXiv references with peer-reviewed versions where possible.
- Avoid online articles as core supporting evidence unless necessary.
- Consider merging redundant contribution bullets.

---

# Suggested Resubmission Strategy

The paper should be reframed around a clearer experimental claim.

A strong revised version could focus on this question:

> Does adding an explicit knowledge graph improve grounded gameplay question answering compared to vector-only or parametric baselines?

To support this claim, the revised paper should first prove the benefit of the KG using a controlled or partially manual KG. Once that is established, the automatic extraction pipeline can be presented as a scalable way to build the KG from gameplay videos.

This would directly answer the reviewers’ main concern: that the current paper tries to prove both the extraction pipeline and the usefulness of the graph at the same time, making the contribution harder to evaluate.
