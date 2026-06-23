from __future__ import annotations

import json
import tempfile
from pathlib import Path

import networkx as nx

from knowledge_loop.inventory import build_inventory_snapshot, persist_inventory_snapshot


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_cache(
    cache_root: Path,
    *,
    video_name: str,
    game: str,
    entity: str,
    chunk_text: str,
    include_lol_fields: bool = False,
) -> None:
    video_dir = cache_root / f"sanitized_build_cache_{video_name}"
    video_dir.mkdir(parents=True, exist_ok=True)
    chunk_id = f"chunk-{video_name.lower()}"
    segment_ref = f"{video_name}_0"

    _write_json(
        video_dir / "kv_store_text_chunks.json",
        {
            chunk_id: {
                "content": chunk_text,
                "tokens": max(1, len(chunk_text) // 4),
                "chunk_order_index": 0,
                "video_segment_id": [segment_ref],
            }
        },
    )
    _write_json(
        video_dir / "kv_store_video_segments.json",
        {
            video_name: {
                "0": {
                    "time": "0-30",
                    "content": chunk_text,
                    "transcript": chunk_text,
                    "frame_times": [0.0, 6.0, 12.0],
                }
            }
        },
    )

    frame = {
        "frame_path": f"/tmp/{video_name}_frame.png",
        "segment_idx": "0",
        "segment_name": "fixture-0-0-30",
        "frame_idx": 0,
        "game": game,
        "entities": [entity],
        "transcript": chunk_text,
        "vlm_output": chunk_text,
    }
    if include_lol_fields:
        frame["main_champ"] = entity
        frame["partners"] = ["Fiora"]

    _write_json(video_dir / "kv_store_video_frames.json", {video_name: {"0_0": frame}})
    _write_json(video_dir / "kv_store_video_path.json", {video_name: f"/tmp/{video_name}.mp4"})

    graph = nx.Graph()
    graph.add_node(
        entity,
        entity_type="PERSON",
        description=chunk_text,
        source_id=chunk_id,
    )
    nx.write_graphml(graph, video_dir / "graph_chunk_entity_relation_clean.graphml")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="knowledge-loop-inventory-") as tmp:
        root = Path(tmp)
        cache_root = root / "cache"
        state_root = root / "state"

        _write_cache(
            cache_root,
            video_name="LoL_Test",
            game="league_of_legends",
            entity="Aatrox",
            chunk_text=(
                "Aatrox rune setup uses Conqueror and item build choices against "
                "Fiora in lane."
            ),
            include_lol_fields=True,
        )
        _write_cache(
            cache_root,
            video_name="Generic_Test",
            game="other",
            entity="Mario",
            chunk_text=(
                "Mario drift mechanics improve track route strategy and timer UI awareness."
            ),
        )

        lol_snapshot = build_inventory_snapshot(
            profile_id="league_of_legends",
            cache_root=cache_root,
        )
        assert lol_snapshot.loop_profile_id == "league_of_legends"
        assert set(lol_snapshot.videos) == {"LoL_Test"}
        lol_topics = {row.topic_key for row in lol_snapshot.topic_facets}
        assert "RUNES::AATROX" in lol_topics
        assert all(row.freshness_days is None for row in lol_snapshot.topic_facets)

        written = persist_inventory_snapshot(lol_snapshot, state_root=state_root)
        assert (state_root / "inventory" / "latest_inventory.json").exists()
        assert len(written) == 2

        generic_snapshot = build_inventory_snapshot(
            profile_id="generic_gameplay",
            cache_root=cache_root,
        )
        assert generic_snapshot.loop_profile_id == "generic_gameplay"
        assert set(generic_snapshot.videos) == {"Generic_Test"}
        generic_topics = {row.topic_key for row in generic_snapshot.topic_facets}
        assert "MECHANIC::MARIO" in generic_topics
        assert all(row.freshness_days is None for row in generic_snapshot.topic_facets)

        with (state_root / "inventory" / "latest_inventory.json").open("r", encoding="utf-8") as f:
            latest = json.load(f)
        assert latest["loop_profile_id"] == "league_of_legends"

    print("inventory fixture test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

