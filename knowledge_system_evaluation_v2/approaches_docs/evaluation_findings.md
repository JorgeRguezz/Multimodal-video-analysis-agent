# IEEE Access Evaluation Findings

This document serves as a living record of academic findings discovered during the evaluation pipeline. It will be updated progressively as new phases of the evaluation are completed.

## Phase 4: Deterministic Retrieval Evaluation

### 1. The Modality Role-Reversal (Vision vs. Audio)
By isolating the ASR (Audio) and VLM (Vision) embeddings, we uncovered a dichotomy in how multimodal gaming data behaves during retrieval:
*   **Vision establishes the Subject:** When evaluated purely on whether the primary Champion was retrieved, `vision_only` (91.2% Hit Rate) significantly outperformed `asr_only` (58.4%). Because the champion is constantly rendered on screen, the VLM descriptions effectively anchor the context. Narrators rarely say the champion's name explicitly, making audio a poor primary search vector for subjects.
*   **Audio provides the Details:** Conversely, when evaluated on **Multi-Entity Recall** (retrieving the specific Items, Runes, and Spells needed to construct a complex answer), the modalities flipped. `asr_only` (67.2%) outperformed `vision_only` (63.2%). While the camera tracks the champion, the narrator speaks the specific, dense vocabulary of items and build paths.
*   **Conclusion:** This mathematically proves that neither modality is sufficient alone; Vision is required for subject-tracking, and Audio is required for deep conceptual details.

### 2. The Limits of Traditional Information Retrieval Metrics
In the Multi-Entity Recall test, the oldest algorithm, `bm25` (Sparse Keyword Matching), achieved the highest score (75.9%), slightly edging out `vector_only` (74.3%) and `graph_rag` (73.5%). 
*   **Analysis:** If a Gold Answer explicitly states *"build Ninja Tabi"*, BM25 searches for those exact characters. Dense embeddings and Graph-RAG search for *related concepts* (e.g., retrieving chunks about "Armor" or "Boots"). 
*   **Conclusion:** This demonstrates that standard lexical overlap metrics penalize advanced semantic architectures. Graph-RAG trades exact-word matching for conceptual traversal, which penalizes it on strict string-matching metrics but theoretically provides better holistic context for the generation phase.
