from __future__ import annotations

import os

from .models import LoopGameKnowledgeProfile


LEAGUE_OF_LEGENDS_PROFILE = LoopGameKnowledgeProfile(
    id="league_of_legends",
    display_name="League of Legends Knowledge Loop",
    extraction_game_id="league_of_legends",
    facets=(
        "identity",
        "abilities",
        "runes",
        "itemization",
        "lane_phase",
        "matchups",
        "macro",
        "teamfighting",
        "pathing",
        "combos",
        "visual_sequences",
    ),
    default_facet="identity",
    facet_keywords={
        "abilities": (
            "ability",
            "abilities",
            "spell",
            "ultimate",
            "cooldown",
            "cooldowns",
            "passive",
        ),
        "runes": (
            "rune",
            "runes",
            "keystone",
            "conqueror",
            "electrocute",
            "first strike",
            "resolve",
            "precision",
            "domination",
            "sorcery",
            "inspiration",
        ),
        "itemization": (
            "item",
            "items",
            "build",
            "boots",
            "buy",
            "purchase",
            "zhonya",
            "hourglass",
            "liandry",
            "blade",
        ),
        "lane_phase": (
            "lane",
            "laning",
            "wave",
            "trade",
            "trading",
            "cs",
            "minion",
            "level 1",
            "level 2",
        ),
        "matchups": (
            "matchup",
            "matchups",
            "versus",
            " vs ",
            "against",
            "counter",
            "counters",
        ),
        "macro": (
            "macro",
            "roam",
            "roaming",
            "objective",
            "objectives",
            "dragon",
            "baron",
            "rotate",
            "rotation",
            "vision",
            "ward",
            "wards",
        ),
        "teamfighting": (
            "teamfight",
            "team fight",
            "fight",
            "fighting",
            "engage",
            "skirmish",
            "5v5",
        ),
        "pathing": (
            "pathing",
            "jungle path",
            "route",
            "clear",
            "gank",
            "ganking",
        ),
        "combos": (
            "combo",
            "combos",
            "animation cancel",
            "flash combo",
        ),
        "visual_sequences": (
            "visible",
            "frame",
            "hud",
            "map",
            "ui",
            "health bar",
            "sequence",
        ),
    },
    topic_prefix_by_facet={
        "identity": "CHAMPION",
        "abilities": "ABILITIES",
        "runes": "RUNES",
        "itemization": "ITEMIZATION",
        "lane_phase": "LANE",
        "matchups": "MATCHUP",
        "macro": "MACRO",
        "teamfighting": "TEAMFIGHT",
        "pathing": "PATHING",
        "combos": "COMBOS",
        "visual_sequences": "VISUAL",
    },
    entity_stopwords=("UNKNOWN", "NONE", "N/A"),
    generic_topic_key="GENERAL::LEAGUE_OF_LEGENDS",
    core_gap_facets=(
        "abilities",
        "runes",
        "itemization",
        "lane_phase",
        "matchups",
        "macro",
        "teamfighting",
        "combos",
    ),
    noisy_topic_tokens=(
        "ALLY_CHAMPION",
        "ENEMY_CHAMPION",
        "E_ABILITY",
        "Q_ABILITY",
        "W_ABILITY",
        "R_ABILITY",
        "FLASH",
        "LOW_HEALTH",
        "HEALTH_BAR",
        "MINIMAP",
        "HUD",
        "UI",
    ),
)


GENERIC_GAMEPLAY_PROFILE = LoopGameKnowledgeProfile(
    id="generic_gameplay",
    display_name="Generic Gameplay Knowledge Loop",
    extraction_game_id="other",
    facets=(
        "identity",
        "mechanics",
        "strategy",
        "actions",
        "environment",
        "ui",
        "progression",
        "visual_sequences",
    ),
    default_facet="identity",
    facet_keywords={
        "mechanics": (
            "mechanic",
            "mechanics",
            "control",
            "controls",
            "drift",
            "jump",
            "aim",
            "timing",
        ),
        "strategy": (
            "strategy",
            "plan",
            "route",
            "positioning",
            "decision",
            "optimal",
        ),
        "actions": (
            "attack",
            "move",
            "moving",
            "turn",
            "collect",
            "use",
            "avoid",
        ),
        "environment": (
            "map",
            "area",
            "track",
            "room",
            "terrain",
            "obstacle",
            "environment",
        ),
        "ui": (
            "ui",
            "hud",
            "menu",
            "score",
            "timer",
            "icon",
        ),
        "progression": (
            "level",
            "stage",
            "checkpoint",
            "unlock",
            "progress",
            "rank",
        ),
        "visual_sequences": (
            "visible",
            "frame",
            "sequence",
            "camera",
            "screen",
        ),
    },
    topic_prefix_by_facet={
        "identity": "ENTITY",
        "mechanics": "MECHANIC",
        "strategy": "STRATEGY",
        "actions": "ACTION",
        "environment": "AREA",
        "ui": "UI",
        "progression": "PROGRESSION",
        "visual_sequences": "VISUAL",
    },
    entity_stopwords=("UNKNOWN", "NONE", "N/A"),
    generic_topic_key="GENERAL::GENERIC_GAMEPLAY",
    core_gap_facets=(
        "identity",
        "mechanics",
        "strategy",
        "actions",
        "environment",
    ),
    noisy_topic_tokens=(
        "UNKNOWN_ENTITY",
        "UI",
        "HUD",
        "SCREEN",
        "CAMERA",
    ),
)


LOOP_PROFILES: dict[str, LoopGameKnowledgeProfile] = {
    LEAGUE_OF_LEGENDS_PROFILE.id: LEAGUE_OF_LEGENDS_PROFILE,
    GENERIC_GAMEPLAY_PROFILE.id: GENERIC_GAMEPLAY_PROFILE,
}

EXTRACTION_GAME_TO_LOOP_PROFILE = {
    "league_of_legends": "league_of_legends",
    "other": "generic_gameplay",
}


def get_default_loop_profile_id() -> str:
    explicit = os.environ.get("KNOWLEDGE_LOOP_PROFILE")
    if explicit:
        return explicit
    video_game = os.environ.get("VIDEO_GAME")
    if video_game:
        return EXTRACTION_GAME_TO_LOOP_PROFILE.get(video_game, video_game)
    return "league_of_legends"


def get_loop_profile(profile_id: str | None = None) -> LoopGameKnowledgeProfile:
    resolved = profile_id or get_default_loop_profile_id()
    try:
        return LOOP_PROFILES[resolved]
    except KeyError as exc:
        supported = ", ".join(sorted(LOOP_PROFILES))
        raise ValueError(
            f"Unsupported loop profile {resolved!r}. Supported values: {supported}"
        ) from exc
