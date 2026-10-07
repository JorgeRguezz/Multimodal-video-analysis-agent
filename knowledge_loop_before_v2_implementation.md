# Knowledge Loop Notes Before V2 Implementation

This note records design clarifications that came up while reviewing the first V1 inventory output. It should be treated as guidance for refining V1 before expanding toward V2.

## 1. Inventory Buckets Are Broad by Design

The Stage 1 inventory currently emits raw `topic_key + facet` buckets. These are not yet curated strategic learning targets.

For example:

```text
RUNES::AATROX
ITEMIZATION::AHRI
MACRO::PYKE
ABILITIES::FIORA
VISUAL::LUX
```

With 38 videos, the system produced about 4000 topic/facet buckets. This is high, but not automatically wrong. The fanout comes from combining many chunks, detected entities, facets, frame metadata, and graph nodes.

The inventory should remain a broad measurement layer. Gap detection should not blindly treat every raw bucket as an actionable gap.

## 2. Add Actionability Filtering Before Gap Selection

Before Stage 2 gap detection selects targets, the system should filter or down-rank noisy buckets.

Likely low-quality buckets include:

- one-video, one-chunk singleton buckets,
- generic or placeholder-like entities such as `ENEMY_CHAMPION`,
- UI/action tokens incorrectly treated as entities, such as `FLASH` or `E_ABILITY`,
- buckets supported only by weak graph extraction and no frame/entity evidence,
- buckets whose facet match is based on weak keyword overlap.

Stage 2 should introduce an “actionable bucket” layer so the supervisor sees a smaller, cleaner candidate set.

## 3. JSON Inventory Is an Export, SQLite Should Become Canonical

`latest_inventory.json` is useful for inspection and debugging, but it should not be the long-term canonical store once the KB grows.

Before scaling toward thousands of videos, inventory data should also be written to SQLite tables such as:

- `inventory_runs`,
- `inventory_videos`,
- `inventory_topic_facets`,
- `inventory_topic_facet_sources`,
- `inventory_warnings`.

The JSON snapshot can remain as a human-readable export, while SQLite becomes the queryable state used by gap detection and future loop stages.

## 4. Weak Redundancy Should Become a Real Signal

The current `weak_redundancy_score` is only the inverse of `source_diversity_score`, so it is convenient but not independently informative.

Before V2, this should be improved to combine:

- low video count,
- low uploader/source diversity when metadata exists,
- too many chunks concentrated in one video,
- no or weak graph support,
- poor spread across segments.

For V1, it is acceptable as a simple under-support signal. For V2, it should become a richer evidence-quality metric.

## 5. Practical Rule

Keep Stage 1 broad and inspectable. Make Stage 2 selective.

The loop should first measure the KB generously, then filter aggressively before asking the agent to choose what to learn next.

