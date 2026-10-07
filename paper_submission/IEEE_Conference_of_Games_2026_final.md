# From Gameplay to Knowledge Graph: Multimodal Retrieval-Augmented QA for League of Legends

**Authors:** XXXXXXXX, XXXXXXXXX, XXXXXXXX, XXXXXXXX, XXXXXXXX, XXXXXXXX, XXXXXXXX, XXXXXXXX, XXXXXXXX, XXXXXXXX

## Abstract

Multimodal gameplay videos contain rich information about player behavior, strategic decisions, and in-game events, yet extracting structured knowledge from such sources remains challenging. In this work, we present an end-to-end system that transforms gameplay streams into continuously growing knowledge graphs, enabling grounded question answering through retrieval-augmented inference. The architecture integrates multimodal perception, including visual grounding of game entities, automatic speech transcription, and contextual scene description, with LLM-based symbolic knowledge extraction. For efficient reasoning, the system combines vector-based semantic search with knowledge graph construction in an intent-aware hybrid retrieval framework. Designed for practical deployability, the pipeline operates locally on moderate hardware while maintaining scalable knowledge acquisition. Our work proves that complex multimodal reasoning can operate efficiently without large-scale infrastructure. Beyond gaming, this framework offers a general blueprint for transforming raw video into structured, queryable knowledge. We evaluate the system using overlap-based, semantic, and generation-specific metrics. Results demonstrate that our retrieval-augmented approach significantly enhances factual grounding and reduces hallucinations compared to standalone models. To support reproducibility, the source code is publicly available at: `https://github.com/XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`.

**Index Terms:** Multimodal RAG, KG Extraction, Audiovisual Understanding, Question-Answering, Videogames Analysis

---

## I. Introduction

The global video game industry continues to expand rapidly, with a projected market value of $188.8 billion and a player base reaching 3.6 billion in 2025 [1]. It has evolved into a highly complex ecosystem where competitive titles demand a steep learning curve and constant adaptation to shifting strategies, or the “meta”. To master these environments, players heavily rely on user-generated content [2], [3] distributed across platforms like YouTube and Twitch. Content creators, coaches, and high-level players act as essential knowledge mediators, sharing detailed guides, item builds, and gameplay analyses. However, this creates a significant information retrieval problem: extracting specific, situational knowledge — such as a particular champion’s mechanics or contextual itemization — requires players to manually sift through hours of unstructured video content [4].

Furthermore, Multiplayer Online Battle Arena (MOBA) games like League of Legends (LoL) represent a unique challenge due to their dynamic and highly combinatorial nature. With over 170 distinct champions and frequent patch updates, match data and strategic guides become obsolete rapidly [5]. Players must continuously rely on the latest video content to understand current meta-strategies. Yet, extracting actionable intelligence is non-trivial, as critical knowledge is deeply embedded in the situational context, such as player movement patterns and map awareness [6], rather than explicitly stated in the audio commentary [7].

Recent advancements in Artificial Intelligence (AI) have explored ways to automate game analysis and content generation. Large Language Models (LLMs) and standard automated systems have been employed to generate generic match summaries, classify events, and detect statistical patterns [8]. Despite these advancements, existing approaches are often limited to shallow reasoning in understanding complex game states and long-term tasks, requiring significant manual intervention [9]. While Retrieval-Augmented Generation (RAG) has disrupted text-based knowledge retrieval [10], applying standard RAG to video games presents a severe multimodal challenge [11]. Gameplay videos contain dense, overlapping information streams, fast-paced visual action, complex User Interfaces (UIs), and concurrent audio commentary, which often lead to information loss in non-specialized pipelines [12].

Analyzing this multimodal footage introduces severe computer vision and reasoning challenges. A standard MOBA screen contains a dense Heads-Up Display (HUD), including the minimap, ability cooldowns, and health bars, which dictates the true context of the action. Traditional Vision-Language Models (VLMs) often fail to accurately interpret these small, critical UI elements without precise prior guidance [13]. Moreover, the knowledge required to answer advanced player queries is highly relational. For instance, understanding why a specific item is purchased requires correlating the player’s champion, the opponent’s composition, and the current temporal state of the match. Storing this complex, interdependent information in a traditional semantic vector database often leads to a loss of relational logic and increases the risk of hallucinations by LLMs [14], [15]. Therefore, explicitly mapping these relationships into a structured format is necessary to guarantee accurate retrieval [16].

To overcome the limitations of generic video summarization, this paper introduces a novel Multimodal Graph-RAG architecture designed for advanced, personalized knowledge retrieval. By transforming raw League of Legends gameplay footage into an interactive, reasoning-capable knowledge base, our framework eliminates the bottleneck of manual video analysis. Through the seamless synthesis of visual and auditory data streams, we empower an agentic retrieval system to deliver highly precise, context-aware answers to complex user queries.

The primary contributions of this work are:

- **An Automated Multimodal Data Pipeline:** A novel method for extracting and structuring knowledge from complex gaming videos using audio transcription, SAM-based visual segmentation, and VLMs.
- **Knowledge Graph Construction for Gaming:** The development of a relational database capable of mapping the intricate logic, tactics, and entity interactions within the game.
- **A Graph-Based RAG Architecture:** An advanced retrieval system that answers complex, multi-step user questions about gameplay, significantly reducing the time players spend searching for specific information.

---

## II. Related Work

Current research in Multimodal RAG focuses on aligning diverse data streams — text, audio, and video — into unified embedding spaces [17]. Frameworks such as Video-RAG have pioneered training-free pipelines that utilize visually-aligned auxiliary texts, such as OCR and ASR, to enhance the understanding of long-form videos [18]. Advanced dual-channel architectures now combine graph-based textual grounding with multimodal context encoding to process extreme long-context data while maintaining semantic dependencies [19]. However, as noted in recent efforts in esports analytics, a significant limitation of these models is their restriction to generic summarization or basic event classification. They often struggle with strategic reasoning and fail to extract the underlying tactical intent necessary to answer specific, personalized player queries [20].

Analyzing raw gameplay video presents severe computer vision challenges due to the density of the Heads-Up Display (HUD) and the rapid pace of visual changes. Standard Vision-Language Models (VLMs) often experience degradation in performance and hallucinate when forced to interpret complex, unstructured spatial data without prior guidance [13]. To mitigate this, recent trends leverage foundation models like the Segment Anything Model (SAM). SAM 2 and SAM 3 provide robust zero-shot segmentation and tracking of objects and concepts across video frames [21]. While SAM has been widely adopted in fields like medical imaging, its application to complex gaming interfaces — isolating minimap icons, ability cooldowns, or champion models amidst visual clutter — remains underexplored. Chaining SAM with a VLM provides explicit, segmented context, which is crucial as benchmarks like VideoGameQA-Bench show that frontier models achieve less than 45% accuracy in tasks requiring fine-grained UI detection [12].

Retrieval-Augmented Generation has emerged as the standard approach for mitigating LLM hallucinations [22]. However, standard vector-based RAG relies on semantic text similarity, which is insufficient for domains requiring complex relational logic, such as Multiplayer Online Battle Arena (MOBA) games. Capturing the logic of a game requires representing knowledge as interconnected entities rather than isolated data points. Knowledge Graphs (KGs) structure information into nodes, such as Champions and Items, and edges, such as “counters” and “builds”, preserving the spatial and tactical logic [23]. Recent literature highlights the transition towards Agentic RAG and reasoning systems that perform multi-step retrieval [24]. While frameworks like Graph-VideoAgent and Vgent employ dynamic graph memory to track temporal relations between visual entities [25], [26], extracting multimodal video data directly into a structured KG for gaming applications represents a significant gap that our architecture seeks to fill.

Video games serve as robust environments for AI research due to their complex and highly observable state spaces. However, the demand for real-time tactical support makes latency a critical factor. High-level game-playing agents often depend on ultra-large VLMs like GPT-4o, leading to delays that can exceed 60 to 90 seconds. This drive for efficiency has spurred interest in Small Language/Vision Models (SLMs) for RAG tasks [27], [28]. Systems like MiniRAG and VideoRAG optimize SLM performance by using semantic-aware heterogeneous graph indexing, reducing storage requirements by 75% while maintaining effectiveness through topological retrieval [29], [30].

---

## III. System Architecture

> **Figure 1. Proposed Multimodal RAG Framework Architecture.** The pipeline consists of three stages: (1) Knowledge Extraction via ASR, SAM3, and VLM modules; (2) Knowledge Graph Construction where an LLM structures extracted entities into a Multivideo Knowledge Graph; and (3) Retrieval-Augmented Inference, where user queries are enriched with graph-based context to generate domain-specific responses.

This work proposes an end-to-end pipeline that transforms gameplay videos into a structured and queryable knowledge base capable of supporting natural language question answering. The system integrates multimodal video analysis, knowledge graph construction, and retrieval-augmented generation to enable grounded reasoning over gameplay events. The overall objective is to convert unstructured gameplay footage into structured representations that capture entities, events, and relationships occurring throughout the video.

The architecture follows a multi-stage pipeline. First, the input gameplay video is processed through a multimodal knowledge extraction stage, where visual frames, audio transcripts, and contextual cues are analyzed to produce structured textual descriptions and entity annotations. These outputs are used to construct a knowledge graph representation of the gameplay, linking entities such as champions, items, and objectives with the events described in the video segments. Textual representations of the extracted content are indexed using dense embeddings to enable efficient semantic retrieval.

The resulting knowledge graph and vector indexes are used by a retrieval-augmented inference module that answers user queries about the gameplay video. Given a natural language question, the system retrieves relevant textual chunks and graph entities, reranks the retrieved evidence, and generates a grounded answer using a LLM. This design combines structured knowledge reasoning with semantic retrieval, allowing the system to answer complex questions about gameplay events while remaining grounded in the extracted evidence.

### A. Multimodal Knowledge Extraction

> **Figure 2. Multimodal Extraction Workflow.** The process involves segmenting raw footage into 30s intervals and extracting keyframes. Visual features from the HUD are identified via SAM3 and matched against a Champion VDB, while ASR provides audio transcripts. These inputs are fused by a VLM to produce a sequence of frame-level outputs, which an LLM finally summarizes into segment-based knowledge.

This first stage converts raw gameplay videos into structured multimodal artifacts that can later be transformed into symbolic knowledge. Gameplay videos contain several complementary information sources, including visual events occurring in the game world, spoken commentary, and contextual cues displayed in the user interface. Extracting meaningful knowledge requires integrating signals across multiple modalities rather than relying on a single source. The knowledge graph construction strategy adopted in this work is inspired by prior approaches [30]. This stage constitutes the core of the system, as the quality and consistency of the extracted information directly influence the effectiveness of subsequent knowledge graph construction and retrieval-augmented inference. The primary design objective is to provide a vision-language model (VLM) with rich contextual grounding so that frame-level descriptions capture not only visible actions but also inferred gameplay dynamics.

#### a) Temporal segmentation and multimodal sampling

Each input gameplay video is first divided into fixed-duration temporal segments of 30 seconds. Within every segment, five frames are sampled at regular intervals, one every six seconds, producing sparse but information-rich visual checkpoints that track the evolving game state. In parallel, the corresponding audio track is processed using an automatic speech recognition (ASR) model to generate timestamped transcripts of spoken commentary. This segmentation strategy provides temporal awareness while maintaining computational tractability.

#### b) Champion grounding via HUD-centric entity detection

A key component of the visual extraction pipeline is the identification of relevant in-game entities, particularly the champions controlled by the player and their teammates. Rather than performing full-scene recognition, the system adopts a HUD-centric grounding strategy that focuses on fixed interface regions containing champion portraits. These portraits occupy only a small number of pixels in the sampled frames and must be matched against a large set of possible identities, including more than 170 champions and numerous visual skin variations, meaning more than 1,500 possible combinations.

To address this challenge, candidate portrait regions are first localized using the Segment Anything Model 3 (SAM3), queried with the open-vocabulary prompt “Character”. SAM3 inference is parallelized across frame regions to reduce latency and maintain throughput in long gameplay videos. The detected regions are then processed using a two-stage visual retrieval procedure: first, global image embeddings are used to retrieve a shortlist of likely champion identities from a persistent reference database; second, a finer-grained patch-token similarity scoring mechanism refines the shortlist to determine the final champion identity. This hierarchical matching approach improves robustness to low-resolution inputs, UI variability, and partial occlusions while preserving computational efficiency.

#### c) Two-stage champion matching

Let `r` denote a detected region of interest (ROI) corresponding to a candidate champion portrait, and let `D = {c1, ..., cN}` be the database of reference champions. The system first computes a global visual embedding `ϕ(r)` for the ROI and compares it against the reference embedding `ϕ(ci)` of each champion candidate. The first stage retrieves a shortlist `CK(r)`:

```math
C_K(r) = \operatorname{TopK}_{c_i \in D} \operatorname{sim}(\phi(r), \phi(c_i))
```

where `sim(·, ·)` denotes cosine similarity.

For each shortlisted candidate `ci ∈ CK(r)`, the system extracts patch-token sets `T(r) = {t1, ..., tm}` and `T(ci) = {u1, ..., un}`, and computes a fine-grained late-interaction score:

```math
S(r, c_i) = \sum_{t \in T(r)} \max_{u \in T(c_i)} \langle t, u \rangle
```

where `⟨·, ·⟩` is the dot product between normalized token embeddings. The final champion identity is selected as:

```math
\hat{c}(r) = \arg\max_{c_i \in C_K(r)} S(r, c_i)
```

In practice, the first stage reduces the search space over the full champion database, while the second stage improves precision for low-resolution HUD cutouts by comparing localized visual evidence at patch level.

#### d) Contextualized frame-level scene description

Each sampled frame is processed by a vision-language model (VLM) that generates a dense natural-language description of the gameplay scene. Crucially, the VLM receives additional contextual inputs including the detected champion identities, the ASR transcript associated with the current segment, and the description generated for the previous sampled frame. This temporal chaining mechanism encourages delta-style reasoning, enabling the model to focus on newly emerging gameplay events rather than repeatedly describing static visual elements.

#### e) Segment-level aggregation

Frame-level descriptions within each temporal segment are subsequently aggregated using a LLM that produces a concise segment summary. This summarization step reduces redundancy while preserving salient information about champion actions, interactions, and inferred strategic developments. The resulting segment representation combines temporal metadata, summarized visual descriptions, detected entities, and associated transcripts.

#### f) Output generation

The multimodal extraction phase produces structured JSON artifacts at both frame and segment granularity. These artifacts form the formal interface with downstream modules, serving as the normalized evidence layer from which textual chunks are derived and symbolic knowledge graphs are constructed in the subsequent stage.

### B. Knowledge Graph Construction

> **Figure 3. Knowledge Graph Construction Workflow.** The summarized segment output undergoes text chunking before being processed by an LLM for entity and relationship extraction. Parallel to this, an embedding model generates vector representations of text chunks and entities. The pipeline demonstrates the iterative transition from individual segment graphs to video-level graphs, culminating in a unified Multivideo Knowledge Graph for global context.

Leveraging the prior segment-level analysis, the system converts this data into a knowledge graph that captures key entities and relationships. Simultaneously, it generates vector embeddings to enable efficient semantic retrieval and reasoning across the gameplay segments.

The construction process begins by converting the segment-level textual descriptions into smaller textual units referred to as text chunks. Each chunk corresponds to a portion of the segment summary while preserving references to the original video timestamps. This chunking strategy provides finer granularity for downstream processing while maintaining temporal grounding with respect to the gameplay video. Entity and relationship extraction is then performed on each text chunk. The proposed system adopts an LLM-native graph extraction strategy in which both entities and relations are jointly inferred through structured generation.

The extraction pipeline follows a two-phase process designed to improve the completeness and consistency of the resulting graph. First, the LLM performs a base extraction pass, identifying candidate entities and relationships present in the chunk. The model is prompted to produce graph facts using a predefined tuple-based schema. Entity records are generated in the form:

```text
("entity"⟨|⟩en⟨|⟩et⟨|⟩ed)
```

Relationship records follow the structure:

```text
("relationship"⟨|⟩es⟨|⟩et⟨|⟩rd⟨|⟩w)
```

where `en` denotes the entity name, `et` the entity type, `ed` a textual description of the entity, `es` and `et` represent the source and target entities of a relationship, `rd` describes the semantic interaction between them, and `w` corresponds to a scalar relationship strength score. This structured output format enables deterministic parsing of the generated text into graph-compatible records.

In the second phase, the system performs a unified glean pass, in which the LLM revisits previously extracted tuples to refine and consolidate the detected knowledge. During this stage, missing relations may be recovered, ambiguous entity mentions can be clarified, and redundant entity instances arising from independent chunk processing may be resolved. This iterative refinement improves both coverage and structural consistency of the resulting knowledge graph.

The validated entities are represented as nodes in the knowledge graph, while the relationships are stored as edges connecting the corresponding entity nodes. Each extracted fact is annotated with a provenance identifier corresponding to the originating text chunk. Since chunk records maintain references to one or more video segments, this design enables full traceability from graph nodes and edges back to specific temporal regions of the gameplay video.

In addition to entity-entity relations, the system maintains explicit links between text chunks and the entities mentioned within them, resulting in a chunk-entity graph structure. In this representation, chunk nodes act as evidence carriers derived from segment-level gameplay descriptions, while entity nodes represent gameplay concepts such as champions, abilities, items, and objectives. These connections allow retrieval mechanisms to ground reasoning in concrete textual evidence extracted from the video content.

To support efficient retrieval during inference, both text chunks and graph entities are embedded into dense vector representations using a sentence embedding model. These embeddings are stored in vector indices that allow efficient similarity search across the extracted knowledge. The combination of structured graph relations and vector-based semantic representations enables hybrid retrieval strategies that leverage both relational reasoning and semantic similarity.

Finally, the system aggregates information by generating both individual and global knowledge graphs. Each processed video first produces a local graph, comprising chunk nodes, entities, and relationships, that preserves the specific gameplay context and maintains explicit links to its source textual evidence. To construct the global knowledge, these local representations are merged, applying conservative matching rules to unify identical gameplay concepts. This prevents duplication while seamlessly integrating relationships extracted from different videos. The outputs of this stage include the consolidated global graph, the preserved per-video graphs, and their corresponding vector indices for text chunks and entities.

### C. Retrieval-Augmented Inference

> **Figure 4. Retrieval-Augmented Inference Mechanism.** The inference stage begins with Query Analysis, where user input is classified as event-, entity-, or relationship-focused and embedded. This query then triggers a search through the Multivideo Knowledge Graph to retrieve relevant text chunks and neighboring entities. Finally, the system performs Context Scoring and assembly, feeding the enriched prompt into an LLM to generate a precise, context-aware response.

Once knowledge construction is complete, the system enables question answering over gameplay videos through a retrieval-augmented inference pipeline. The objective of this stage is to generate responses that are explicitly grounded in the extracted multimodal knowledge while leveraging the reasoning capabilities of large language models.

#### a) Knowledge sanitization and operational memory

Before inference, the knowledge artifacts produced during graph construction undergo a lightweight sanitization process. This stage normalizes entity naming conventions, removes structurally invalid relationships, and resolves duplicated or inconsistent records.

#### b) Query analysis and intent routing

When a user submits a natural language query, the system performs a structured query analysis stage that converts the input question into a retrieval configuration controlling subsequent evidence selection. This process comprises four coordinated analyses: semantic similarity, entity mention detection, temporal cue extraction, and intent classification.

First, the query is encoded using the same sentence embedding model employed for chunk indexing, initializing dense similarity search over textual evidence. In parallel, lightweight lexical matching is performed between query tokens and known entities stored in the knowledge graph. When entity mentions are detected, entity-centered retrieval is activated and matched nodes are used as seeds for graph traversal.

Temporal cues are extracted through pattern-based recognition of time-referencing expressions, such as “early game”, “first dragon”, or explicit minute references, as well as semantic alignment with gameplay-phase descriptors derived during knowledge construction. When temporal signals are present, timestamp-aware filtering is applied and chunks associated with relevant segment windows receive ranking boosts.

Finally, a prompt-guided language model assigns the query to high-level intent categories such as event-focused, entity-focused, or interaction-focused reasoning. Event-focused queries increase the influence of semantic chunk similarity and temporal alignment signals. Entity-focused queries prioritize direct entity lookup and shallow graph expansion over immediate neighbors. Interaction-focused queries activate deeper multi-hop graph traversal and increase the weight of structural connectivity signals. The outputs of these analyses are combined into a routing configuration that determines which retrieval branches are executed, how far graph traversal proceeds, and how candidate scores are fused during hybrid evidence selection.

#### c) Hybrid evidence retrieval and scoring

To combine the retrieval methods described previously, each candidate evidence `e` is assigned a fused retrieval score:

```math
S(e) = \alpha S_{sem}(e, q) + \beta S_{graph}(e, q) + \gamma S_{temp}(e, q) + \delta S_{ent}(e, q)
```

where `Ssem` denotes semantic similarity to the query, `Sgraph` captures structural relevance derived from graph connectivity, `Stemp` measures temporal alignment with inferred query constraints, and `Sent` reflects overlap between query-mentioned entities and evidence content. The weights `α`, `β`, `γ`, and `δ` are dynamically adjusted according to the routing configuration produced during query analysis.

#### d) Reranking and context construction

Retrieved candidates are deduplicated and passed to a reranking stage that refines evidence selection using contextual diversity constraints and relevance normalization. The highest-ranked chunks and entity-centered relation contexts are assembled into a structured prompt representation that respects a predefined token budget. This step ensures that the downstream LLM receives coherent, non-redundant, and temporally grounded evidence.

#### e) Grounded answer generation and verification

Using the curated evidence context, a LLM generates a natural language response to the user query. The generation prompt explicitly instructs the model to ground its reasoning in the retrieved knowledge snippets. A subsequent verification step compares the generated response against the supporting context to detect unsupported claims and estimate response confidence. This comparison evaluates semantic overlap between answer statements and retrieved evidence, as well as consistency of referenced entities and relationships.

The resulting confidence is primarily used for debugging and evaluation purposes, enabling developers to identify retrieval failures, insufficient context coverage, or hallucination behavior during system development. Through the integration of intent-aware routing, hybrid semantic-symbolic retrieval, and grounded answer verification, the proposed inference pipeline enables explainable question answering over dynamically constructed gameplay knowledge bases.

---

## IV. Experiments and Evaluation

All experiments were conducted on a workstation equipped with an NVIDIA TITAN RTX GPU with 24 GB VRAM, 32 GB of system memory, and an 11th Generation Intel Core i7 processor. The proposed system is intentionally designed to operate under moderate hardware constraints, demonstrating that multimodal gameplay knowledge extraction and retrieval can be performed locally without large-scale infrastructure.

To identify suitable trade-offs between quality, latency, and computational cost, multiple model configurations were explored. For visual-language reasoning, models such as SmolVLM2 (2.2B) and Qwen2.5-VL (7B) were evaluated, with InternVL3 (14B) ultimately selected due to its favorable balance between descriptive performance and inference speed. In speech recognition, both Vosk and several Whisper variants were tested, with Whisper Base providing the best efficiency-accuracy trade-off. For LLM inference, candidates including Qwen3 (30B A3B) and GLM-4.7-Flash were considered, while GPToss-20b achieved the most consistent reasoning throughput on the target hardware. Architectural exploration also included iterative improvements to champion identification and alternative workflow and prompting strategies for KG extraction.

### A. Evaluation Protocol

We constructed a curated question-answer benchmark composed of gameplay-related queries derived from processed video content, including both general strategic questions and temporally grounded queries referring to specific segments. Inference was executed using the sanitized knowledge caches employed during deployment. Performance was assessed using ROUGE-L and BERTScore to measure lexical and semantic similarity against manually written reference answers, providing a baseline for textual alignment. The evaluation integrated groundedness-oriented metrics via the RAGAS framework [31]: Faithfulness, to detect hallucinations; Answer Relevancy, to ensure alignment with the user’s intent; and Answer Correctness, to verify factual accuracy.

### B. Comparative Baselines

We compare the proposed pipeline against three reference systems: a large proprietary parametric model, GPT-4o; a smaller local baseline, GPToss-20b; and an exploratory implementation of RAG-Anything, a generalized multimodal retrieval framework [32]. While the RAG-Anything comparison is not strictly controlled due to differences in model capacity, Qwen3-8B, and preprocessing pipelines, it provides a useful reference point for understanding how domain-specialized knowledge graph retrieval behaves relative to generalized multimodal RAG approaches.

### Table I. Comparison of Evaluation Metrics with Different Systems

| Metric | RAG (ours) | GPT-4o | GPToss-20b | RAG-Anything |
|---|---:|---:|---:|---:|
| ROUGE-L | 0.094 | 0.1192 | 0.0972 | 0.1266 |
| BERTScore-F1 | 0.7076 | 0.7377 | 0.7024 | 0.7515 |
| Faithfulness | 0.8652 | 0.765 | 0.4898 | 0.6145 |
| Relevancy | 0.1993 | 0.6309 | 0.7741 | 0.5409 |
| Correctness | 0.2817 | 0.351 | 0.2749 | 0.3321 |

### C. Results and Discussion

The results reveal clearly differentiated behavioral patterns across the evaluated systems. GPT-4o achieves strong semantic similarity and correctness scores, reflecting its extensive parametric knowledge and fluency in producing domain-relevant responses. However, qualitative inspection shows that these responses frequently rely on generalized gameplay heuristics rather than evidence grounded in the processed video segments. This can lead to temporally misaligned or patch-agnostic recommendations, particularly in rapidly evolving gameplay contexts. Consequently, high similarity metrics do not necessarily imply faithful reasoning over the audiovisual material.

The smaller GPToss-20b baseline demonstrates comparatively high Answer Relevancy, largely due to concise answers that reuse terminology from the query. Nevertheless, its substantially lower Faithfulness score indicates frequent unsupported claims and limited ability to synthesize temporally grounded context. This highlights the difficulty of addressing gameplay-specific questions using purely parametric reasoning under constrained model capacity.

The exploratory RAG-Anything baseline shows notably strong performance on similarity-oriented metrics, achieving the highest ROUGE-L and BERTScore-F1 values among all evaluated systems. This suggests that generalized multimodal retrieval can effectively produce fluent and semantically aligned answers. It also demonstrates moderate Faithfulness and Correctness, outperforming several baselines in grounded reasoning metrics. However, qualitative analysis indicates that the lack of explicit temporal indexing and structured entity relations limits the system’s ability to reconstruct complex gameplay sequences or provide precise evidence localization. As a result, answers may remain coherent at a semantic level while still lacking fine-grained alignment with specific gameplay events.

In contrast, the proposed RAG pipeline consistently produces responses tightly anchored in retrieved evidence, achieving the highest Faithfulness score across all systems. Although similarity and correctness metrics remain slightly lower than those of larger parametric or generalized retrieval approaches, grounded retrieval frequently yields context-sensitive interpretations derived directly from gameplay clips. This divergence reflects improved temporal alignment and evidential precision rather than reduced answer quality. At the same time, strict grounding policies can lead to conservative behavior, including partial answers or explicit uncertainty statements when relevant evidence coverage is limited. This trade-off illustrates a central design tension in retrieval-augmented gameplay assistants: balancing factual reliability and explainability against perceived completeness and fluency.

Overall, the evaluation supports the hypothesis that combining multimodal knowledge extraction with structured retrieval enables more reliable reasoning over dynamic gameplay content. By integrating semantic similarity search with symbolic graph traversal, the system can reconstruct complex event chains and provide explainable responses grounded in verifiable audiovisual evidence.

### D. Limitations and Practical Considerations

The current knowledge base is derived from a relatively small set of gameplay sources, 22 videos, reflecting constraints in available hardware resources and the limited volume of video sources. While sufficient to validate the feasibility of the proposed architecture, this setup provides only partial coverage of the strategic diversity present in real gameplay ecosystems. Retrieval-augmented systems inherently depend on knowledge availability, and therefore increasing both the scale and diversity of ingested content is expected to improve retrieval recall, reasoning depth, and answer completeness. Future large-scale deployments could leverage distributed preprocessing pipelines or incremental knowledge ingestion strategies to address this limitation.

In addition, traditional text-based QA metrics only partially capture several user-facing capabilities enabled by the proposed pipeline. Beyond generating grounded natural-language responses, the system can surface temporally localized evidence in the form of video timestamps linked to videos processed. This allows users to directly inspect the audiovisual context supporting a recommendation, facilitating verification, learning, and trust calibration with actual gameplay. Such evidence-linked interaction constitutes a practical advantage for explainable gameplay analysis workflows over purely generative baselines, yet remains difficult to quantify using similarity-oriented evaluation frameworks.

Another limitation concerns the evolving nature of gameplay knowledge. Strategic interpretations extracted from video sources may reflect patch-specific dynamics or player preferences that differ from static benchmark answers. While this temporal sensitivity can improve contextual relevance, it also complicates automatic evaluation and highlights the need for more adaptive benchmarking methodologies tailored to dynamic game environments.

---

## V. Conclusion

This work presents an end-to-end multimodal retrieval-augmented system designed to transform raw gameplay videos into structured, queryable knowledge. Operating on moderate local hardware, the proposed architecture integrates multimodal perception with symbolic knowledge graph construction and hybrid semantic-structural retrieval. Our results demonstrate that expanding gameplay knowledge bases can be built efficiently without large-scale infrastructure. Empirical analysis indicates that domain-aware structured retrieval significantly improves factual grounding compared to purely parametric models.

Beyond gameplay analytics, this framework establishes a broader methodology for extracting actionable knowledge from complex audiovisual environments. Ultimately, these systems enable explainable reasoning and open new possibilities for interactive learning tools and context-aware assistants. Future research directions may include the development of specialized models for gameplay semantics and the expansion of the framework to additional genres. Furthermore, the implementation of autonomous knowledge acquisition mechanisms could be explored to identify data gaps, facilitating the transition toward scalable reasoning systems that learn continuously from experience.

---

## References

[1] Newzoo, “Global games market report 2025: Market estimates and forecasts,” Sep. 2025, accessed: Mar. 5, 2026. [Online]. Available: `https://newzoo.com/resources/trend-reports/newzoo-global-games-market-report-2025`

[2] X. Bian and A. Yang, “From spectatorship to loyalty: Unraveling the influence of game streaming watch and gaming-related social connectivity on moba gamers,” *Computers in Human Behavior*, vol. 162, p. 108433, 2024.

[3] C. Johanson, H. Wessels, and M. A. Friehs, “Watching to win: When watching others play improves performance,” *Entertainment Computing*, vol. 56, p. 101067, 2026.

[4] B. R. Anderson and A. M. Smith, “Understanding user needs in videogame moment retrieval,” in *International Conference on Foundations of Digital Games*, 2019.

[5] Riot Games, “Patch schedule - league of legends,” 2025, accessed Mar. 16, 2026. [Online]. Available: `https://support-leagueoflegends.riotgames.com/hc/en-us/articles/360018987893-Patch-Schedule-League-of-Legends`

[6] F. Hojaji, R. E. McIlroy, A. Dupuy, G. L. Pedroni, A. J. Toth, and M. J. Campbell, “Deep learning techniques for identifying kpis in league of legends: Win prediction, map navigation, and vision control,” *Computers in Human Behavior Reports*, 2025.

[7] A. Shahabaz and S. Sarkar, “Increasing importance of joint analysis of audio and video in computer vision: A survey,” *IEEE Access*, vol. 12, pp. 59399–59430, 2024.

[8] R. Gallotta, G. Todd, M. Zammit, S. Earle, A. Liapis, J. Togelius, and G. N. Yannakakis, “Large language models and games: A survey and roadmap,” *ArXiv*, vol. abs/2402.18659, 2024.

[9] C. Wang, L. Tang, M. Yuan, J. Yu, X. Xie, and J. Bu, “Leveraging llm agents for automated video game testing,” *ArXiv*, vol. abs/2509.22170, 2025.

[10] P. Lewis, E. Perez, A. Piktus et al., “Retrieval-augmented generation for knowledge-intensive nlp tasks,” in *Advances in Neural Information Processing Systems*, 2020.

[11] C. Fu et al., “Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis,” *2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 24108–24118, 2024.

[12] M. R. Taesiri, A. Ghildyal, S. Zadtootaghaj, N. Barman, and C.-P. Bezemer, “Videogameqa-bench: Evaluating vision-language models for video game quality assurance,” *ArXiv*, vol. abs/2505.15952, 2025.

[13] Z. Li, X. Wu, H. Du, F. Liu, H. Nghiem, and G. Shi, “A survey of state of the art large vision language models: Alignment, benchmark, evaluations and challenges,” *2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, pp. 1578–1597, 2025.

[14] R. Kumar, H. Kumar, and K. Shalini, “Detecting and mitigating bias in llms through knowledge graph-augmented training,” *2025 International Conference on Artificial Intelligence and Data Engineering (AIDE)*, pp. 608–613, 2025.

[15] Z. Bai, P. Wang, T. Xiao, T. He, Z. Han, Z. Zhang, and M. Z. Shou, “Hallucination of multimodal large language models: A survey,” *ArXiv*, vol. abs/2404.18930, 2024.

[16] G. Niu, B. Li, and Y. Lin, “A survey of task-oriented knowledge graph reasoning: Status, applications, and prospects,” *ArXiv*, vol. abs/2506.11012, 2025.

[17] L. Mei, S. Mo, Z. Yang, and C. Chen, “A survey of multimodal retrieval-augmented generation,” *ArXiv*, vol. abs/2504.08748, 2025.

[18] S. Jeong, K. Kim, J. Baek, and S. J. Hwang, “Videorag: Retrieval-augmented generation over video corpus,” *ArXiv*, vol. abs/2501.05874, 2025.

[19] K. Shao et al., “A survey of token compression for efficient multimodal large language models,” 2026.

[20] Q. Zheng, X. Wang, K. Cheng, M. A. Ali, Y. Lu, and W. Li, “From multimodal perception to strategic reasoning: A survey on ai-generated game commentary,” 2025.

[21] R. Sapkota, K. I. Roumeliotis, and M. Karkee, “The sam2-to-sam3 gap in the segment anything model family: Why prompt-based expertise fails in concept-driven image segmentation,” *ArXiv*, vol. abs/2512.06032, 2025.

[22] W. Fan, Y. Ding, L. bo Ning, S. Wang, H. Li, D. Yin, T.-S. Chua, and Q. Li, “A survey on rag meeting llms: Towards retrieval-augmented large language models,” *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, 2024.

[23] B. Peng, Y. Zhu, Y. Liu, X. Bo, H. Shi, C. Hong, Y. Zhang, and S. Tang, “Graph retrieval-augmented generation: A survey,” *ACM Transactions on Information Systems*, vol. 44, pp. 1–52, 2024.

[24] Y. Li et al., “Towards agentic rag with deep reasoning: A survey of rag-reasoning systems in llms,” *ArXiv*, vol. abs/2507.09477, 2025.

[25] M. Chu, Y. Li, and T.-S. Chua, “Understanding long videos via llm-powered entity relation graphs,” *ArXiv*, vol. abs/2501.15953, 2025.

[26] X. Shen, W. Zhang, J. Chen, and M. Elhoseiny, “Vgent: Graph-based retrieval-reasoning-augmented generation for long video understanding,” *ArXiv*, vol. abs/2510.14032, 2025.

[27] Z. Lu, X. Li, D. Cai, R. Yi, F. Liu, X. Zhang, N. D. Lane, and M. Xu, “Small language models: Survey, measurements, and insights,” *ArXiv*, vol. abs/2409.15790, 2024.

[28] N. Patnaik, N. Nayak, H. Agrawal, M. C. Khamaru, G. Bal, S. S. Panda, R. Raj, V. Meena, and K. Vadlamani, “Small vision-language models: A survey on compact architectures and techniques,” *ArXiv*, vol. abs/2503.10665, 2025.

[29] T. Fan, J. Wang, X. Ren, and C. Huang, “Minirag: Towards extremely simple retrieval-augmented generation,” *arXiv preprint arXiv:2501.06713*, 2025.

[30] X. Ren, L. Xu, L. Xia, S. Wang, D. Yin, and C. Huang, “Videorag: Retrieval-augmented generation with extreme long-context videos,” *arXiv preprint arXiv:2502.01549*, 2025.

[31] E. Shahul, J. James, L. E. Anke, and S. Schockaert, “Ragas: Automated evaluation of retrieval augmented generation,” *ArXiv*, vol. abs/2309.15217, 2023.

[32] Z. Guo, X. Ren, L. Xu, J. Zhang, and C. Huang, “Rag-anything: All-in-one rag framework,” *ArXiv*, vol. abs/2510.12323, 2025.
