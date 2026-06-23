from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get("KNOWLEDGE_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()

SANITIZED_CACHE_ROOT = PROJECT_ROOT / "knowledge_sanitization" / "cache"
DEFAULT_STATE_ROOT = PROJECT_ROOT / "knowledge_loop_state"

SANITIZED_BUILD_PREFIX = "sanitized_build_cache_"
SANITIZED_BUILD_GLOB = f"{SANITIZED_BUILD_PREFIX}*"

INVENTORY_DIR_NAME = "inventory"
GAPS_DIR_NAME = "gaps"
SEARCH_PLANS_DIR_NAME = "search_plans"
SEARCH_RUNS_DIR_NAME = "search_runs"
CANDIDATE_DOWNLOADS_DIR_NAME = "downloads"
SCOUT_RUNS_DIR_NAME = "scout_runs"
SELECTIONS_DIR_NAME = "selections"
