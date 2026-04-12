from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .core import (
    RuleModifiers,
    build_gold_plan_for_row,
    compile_environment_runtime,
)


# ---------------------------------------------------------------------------
# Scarcity sensitivity tagging
# ---------------------------------------------------------------------------

def tag_scarcity_sensitivity(
    row: dict[str, Any],
    env_runtime: dict[str, Any],
    *,
    test_k: int = 12,
) -> bool:
    """Return True if this mission's gold plan changes under scarcity rules.

    Compares gold plans with standard vs scarcity RuleModifiers. Symbolic
    ``bad_quality`` conditions are concretized to *test_k* (default 12,
    the maximum) so all cyclic revisitations are captured.
    """
    zones = env_runtime["zones"]
    concrete = dict(row)
    for zone in zones:
        if str(concrete[f"{zone}_condition"]) == "bad_quality":
            concrete[f"{zone}_condition"] = f"bad_quality_{test_k}"

    gold_standard = build_gold_plan_for_row(concrete, env_runtime, rules=None)
    gold_scarcity = build_gold_plan_for_row(
        concrete, env_runtime, rules=RuleModifiers(scarcity=True),
    )
    return gold_standard != gold_scarcity


def build_l1_structural_key(row: dict[str, Any]) -> str:
    """Build a structural class key for an L1 mission.

    Format: ``{target_zone}_{chain|ring}``  (8 classes for the 4-zone ABCD env).
    """
    target = str(row["target_zone"])
    condition = str(row[f"{target}_condition"])
    cond_type = "chain" if condition == "low_resource" else "ring"
    return f"{target}_{cond_type}"


# ---------------------------------------------------------------------------
# Pool allocation
# ---------------------------------------------------------------------------

@dataclass
class PoolAllocation:
    """Result of splitting missions into non-overlapping pools.

    Attributes:
        pool_a: Missions used only in original train phase (never re-used).
        pool_b: Missions used in probe phases — reused across original_probe,
                adapted_probe, and retention_post_reversal for paired comparison.
        adapted_train: Scarcity-affected missions from pool_a for adapted training.
        unused_l1: L1 missions excluded from both pools (high redundancy).
    """
    pool_a: list[dict[str, Any]]
    pool_b: list[dict[str, Any]]
    adapted_train: list[dict[str, Any]]
    unused_l1: list[dict[str, Any]]


def allocate_pools(
    df: pd.DataFrame,
    env_runtime: dict[str, Any],
    *,
    seed: int = 42,
    l1_per_structural_class: int = 2,
    l2_affected_in_a: int = 6,
    l2_unaffected_in_a: int = 12,
    l3_affected_in_a: int = 6,
    l3_unaffected_in_a: int = 6,
) -> PoolAllocation:
    """Split the 152-mission state dataset into Pool A and Pool B.

    Design principles:
      - Pool A (train-only): used once for original training phase.
      - Pool B (probe-reusable): used in original_probe, adapted_probe, and
        retention_post_reversal for paired within-mission comparisons.
      - Pool A ∩ Pool B = ∅  (zero contamination).
      - L1 missions are highly redundant (9 per structural class); only a
        small sample enters Pool A, the rest are unused.
      - Adapted training draws scarcity-affected missions from Pool A.

    Default allocation yields:
      - Pool A: 16 L1 + 18 L2 + 12 L3 = 46 missions
      - Pool B: 0 L1  + 30 L2 + 20 L3 = 50 missions
      - Adapted train: 12 scarcity-affected from Pool A
      - Unused: 56 L1

    Args:
        df: Full dataset (152 rows for 'state' completeness).
        env_runtime: Compiled environment runtime.
        seed: RNG seed for reproducible sampling.
        l1_per_structural_class: L1 missions sampled per structural class for Pool A.
        l2_affected_in_a: Scarcity-affected L2 missions allocated to Pool A.
        l2_unaffected_in_a: Scarcity-unaffected L2 missions allocated to Pool A.
        l3_affected_in_a: Scarcity-affected L3 missions allocated to Pool A.
        l3_unaffected_in_a: Scarcity-unaffected L3 missions allocated to Pool A.

    Returns:
        PoolAllocation with pool_a, pool_b, adapted_train, unused_l1.
    """
    rng = random.Random(seed)
    rows = df.to_dict(orient="records")

    # Tag all missions
    for row in rows:
        row["_scarcity_affected"] = tag_scarcity_sensitivity(row, env_runtime)
        row["_goal_level"] = int(row["goal_level"])

    l1 = [r for r in rows if r["_goal_level"] == 1]
    l2 = [r for r in rows if r["_goal_level"] == 2]
    l3 = [r for r in rows if r["_goal_level"] == 3]

    # ── L1: sample 2 per structural class (8 classes → 16 missions) ──
    l1_by_key: dict[str, list[dict]] = {}
    for r in l1:
        key = build_l1_structural_key(r)
        l1_by_key.setdefault(key, []).append(r)

    pool_a_l1: list[dict] = []
    unused_l1_rows: list[dict] = []
    for key in sorted(l1_by_key.keys()):
        group = list(l1_by_key[key])
        rng.shuffle(group)
        pool_a_l1.extend(group[:l1_per_structural_class])
        unused_l1_rows.extend(group[l1_per_structural_class:])

    # ── L2: split by scarcity sensitivity ────────────────────────────
    l2_affected = [r for r in l2 if r["_scarcity_affected"]]
    l2_unaffected = [r for r in l2 if not r["_scarcity_affected"]]
    rng.shuffle(l2_affected)
    rng.shuffle(l2_unaffected)

    pool_a_l2 = l2_affected[:l2_affected_in_a] + l2_unaffected[:l2_unaffected_in_a]
    pool_b_l2 = l2_affected[l2_affected_in_a:] + l2_unaffected[l2_unaffected_in_a:]

    # ── L3: split by scarcity sensitivity ────────────────────────────
    l3_affected = [r for r in l3 if r["_scarcity_affected"]]
    l3_unaffected = [r for r in l3 if not r["_scarcity_affected"]]
    rng.shuffle(l3_affected)
    rng.shuffle(l3_unaffected)

    pool_a_l3 = l3_affected[:l3_affected_in_a] + l3_unaffected[:l3_unaffected_in_a]
    pool_b_l3 = l3_affected[l3_affected_in_a:] + l3_unaffected[l3_unaffected_in_a:]

    # ── Assemble pools ───────────────────────────────────────────────
    pool_a = pool_a_l1 + pool_a_l2 + pool_a_l3
    pool_b = pool_b_l2 + pool_b_l3

    # Adapted train: scarcity-affected from Pool A (all levels)
    adapted_train = [r for r in pool_a if r["_scarcity_affected"]]

    return PoolAllocation(
        pool_a=pool_a,
        pool_b=pool_b,
        adapted_train=adapted_train,
        unused_l1=unused_l1_rows,
    )


def summarize_pool_allocation(alloc: PoolAllocation) -> dict[str, Any]:
    """Return a summary dict of pool sizes and composition."""

    def _level_counts(rows: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in rows:
            key = f"L{r.get('_goal_level', r.get('goal_level', '?'))}"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def _scarcity_count(rows: list[dict]) -> int:
        return sum(1 for r in rows if r.get("_scarcity_affected", False))

    pool_a_ids = {r.get("mission_id") for r in alloc.pool_a}
    pool_b_ids = {r.get("mission_id") for r in alloc.pool_b}

    return {
        "pool_a_total": len(alloc.pool_a),
        "pool_a_by_level": _level_counts(alloc.pool_a),
        "pool_a_scarcity_affected": _scarcity_count(alloc.pool_a),
        "pool_b_total": len(alloc.pool_b),
        "pool_b_by_level": _level_counts(alloc.pool_b),
        "pool_b_scarcity_affected": _scarcity_count(alloc.pool_b),
        "adapted_train_total": len(alloc.adapted_train),
        "unused_l1_total": len(alloc.unused_l1),
        "overlap_count": len(pool_a_ids & pool_b_ids),
    }
