"""Task-specific split allocation for the GCSG benchmark tasks.

Provides split allocation for the 11-task benchmark design plus adaptation
sub-pool construction for interleaved training/probing schedules.

Named split options used by the 11-task benchmark:

    random-small       — T1 / T2 / T7 / T8  24 train / 24 probe
                         L1 (8/8)  L2 (8/8)  L3 (8/8)

    accelerated-small  — T3                  12 train / 24 probe
                         L1 (3/8)  L2 (4/8)  L3 (5/8)
                         Half training exposure; L1 train has 3 unique golds
                         from distinct structural classes (no contrastive pairs).

    goalwise-small     — T4                  24 train / 24 probe
                         L1 (8/0)  L2 (16/0)  L3 (0/24)

    holdout-small      — T5                  24 train / 24 probe
                         L1 (8/7)  L2 (8/8)  L3 (8/9)
                         Train rows satisfy NO holdout criterion.
                         Probe rows satisfy ≥ 1 holdout criterion.
                         Holdouts: target_is_biomass, geosphere_is_bad_quality.
                         L1 probe = 7 (geosphere_chain has no holdout variant).
                         L3 probe = 9 to compensate and keep total probe at 24.

    combo-holdout      — T6                  24 train / 24 probe
                         L1 (8/0)  L2 (8/8)  L3 (8/16)
                         Train: same-type combos (BB/LL at L2; BBB/LLL at L3).
                         Probe: cross-type combos (BL/LB at L2; BBL/LBB/BLB/LBL at L3).
                         Canonical-only — read from t6_split CSV column.

    random-large       — T10 / T11           32 train / 24 probe
                         L1 (8/0)  L2 (8/8)  L3 (16/16)
                         For T10 the scarcity-affected gold keys are split
                         evenly between train and probe.

    transfer-probe     — T9 EFGHI            0 train / 32 probe
                         L1 (0/8)  L2 (0/8)  L3 (0/8)
                         Pure probe split for the transfer environment.

Structural rules enforced (per G-SRCG_Benchmark_Plan.md §Task Design):

    1. L1 Train — exactly 3 contrastive pairs (3 base sequences with
       2 distractor variants each = 6 missions) plus 2 independent single
       sequences, yielding 5 unique gold sequences over 8 missions.
       This teaches Distractor Invariance natively.

    2. L1 Probe — exhaustive evaluation of all 8 L1 core topologies
       (4 target zones × 2 primitive branches). One mission per unique
       gold sequence, using a different distractor variant than train.

    3. L2 / L3 / all L2+ probe pools — strictly globally unique gold
       sequences. No gold overlaps internally and Train ∩ Probe = ∅.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .pools import build_l1_structural_key, tag_scarcity_sensitivity


# ---------------------------------------------------------------------------
# Holdout predicates (TODO 2 — T4 uses two of these)
# ---------------------------------------------------------------------------

HOLDOUT_PREDICATES: dict[str, Any] = {
    "target_is_biomass":       lambda r: str(r.get("target_zone", "")) == "biomass",
    "geosphere_is_bad_quality": lambda r: str(r.get("geosphere_condition", "")).startswith("bad_quality"),
    # Legacy predicates retained only so older split_option names still resolve.
    "geosphere_is_low_resource": lambda r: str(r.get("geosphere_condition", "")) == "low_resource",
    "reservoir_is_bad_quality":  lambda r: str(r.get("reservoir_condition", "")).startswith("bad_quality"),
}

# Active holdouts for Task 4 ("holdout-small").
T4_HOLDOUT_CRITERIA = ["target_is_biomass", "geosphere_is_bad_quality"]

# Split option -> (base_split_type, holdout_criteria_list)
SPLIT_OPTION_SPECS: dict[str, tuple[str, list[str]]] = {
    # 11-task named aliases (active).
    "random-small":       ("random_small",       []),
    "accelerated-small":  ("accelerated_small",  []),
    "goalwise-small":     ("goalwise_small",     []),
    "holdout-small":      ("holdout_small",      T4_HOLDOUT_CRITERIA),
    "combo-holdout":      ("combo_holdout",      []),
    "random-large":       ("random_large",       []),
    "transfer-probe":     ("transfer_probe",     []),
}

# K-severity range options (per-task quiz runtime — untouched by this refactor).
K_SEVERITY_OPTIONS: dict[str, dict[str, dict[str, int]]] = {
    "no_k_holdout":       {"train": {"min": 4, "max": 10}, "test": {"min": 4, "max": 10}},
    "some_k_holdout":     {"train": {"min": 4, "max": 8},  "test": {"min": 4, "max": 12}},
    "complete_k_holdout": {"train": {"min": 4, "max": 8},  "test": {"min": 9, "max": 12}},
    "drastic_k_holdout":  {"train": {"min": 4, "max": 8},  "test": {"min": 11, "max": 14}},
}

K_SEVERITY_OPTIONS_ADAPTATION: dict[str, dict[str, dict[str, int]]] = {
    "no_k_holdout":       {"train": {"min": 6, "max": 12}, "test": {"min": 6, "max": 12}},
    "some_k_holdout":     {"train": {"min": 6, "max": 10}, "test": {"min": 6, "max": 14}},
    "complete_k_holdout": {"train": {"min": 6, "max": 10}, "test": {"min": 11, "max": 14}},
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TaskSplit:
    """Result of a train/probe split allocation."""
    train_rows: list[dict[str, Any]]
    probe_rows: list[dict[str, Any]]
    unused_rows: list[dict[str, Any]]
    split_option: str
    summary: dict[str, Any]


@dataclass
class AdaptationPools:
    """Sub-pools for an adaptation study with interleaved schedule."""
    zero_shot_probe_rows: list[dict[str, Any]]
    train_blocks: list[list[dict[str, Any]]]
    probe_blocks: list[list[dict[str, Any]]]
    final_probe_rows: list[dict[str, Any]]
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _matches_any_holdout(row: dict[str, Any], criteria: list[str]) -> bool:
    return any(HOLDOUT_PREDICATES[c](row) for c in criteria)


def _level_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        lvl = r.get("_goal_level", r.get("goal_level", "?"))
        counts[f"L{lvl}"] = counts.get(f"L{lvl}", 0) + 1
    return dict(sorted(counts.items()))


def _scarcity_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for r in rows if r.get("_scarcity_affected", False))


def _unique_gold_count(rows: list[dict[str, Any]]) -> int:
    return len({str(r.get("gold_action_sequence", "")) for r in rows})


def _group_by_gold(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        key = str(r.get("gold_action_sequence", ""))
        groups.setdefault(key, []).append(r)
    return groups


def _verify_disjoint(train_rows: list[dict[str, Any]],
                     probe_rows: list[dict[str, Any]]) -> None:
    """Assertion: Train and Probe gold sequences must be globally disjoint (L2/L3 rule).

    L1 rows are exempt — their contrastive-pair structure is verified separately
    by _sample_l1_contrastive().
    """
    train_l2_l3_golds = {
        str(r["gold_action_sequence"])
        for r in train_rows
        if int(r.get("_goal_level", r.get("goal_level", 0))) >= 2
    }
    probe_l2_l3_golds = {
        str(r["gold_action_sequence"])
        for r in probe_rows
        if int(r.get("_goal_level", r.get("goal_level", 0))) >= 2
    }
    overlap = train_l2_l3_golds & probe_l2_l3_golds
    if overlap:
        raise AssertionError(
            f"Global Sequence Uniqueness violated — L2/L3 gold keys present "
            f"in BOTH train and probe: {sorted(overlap)[:3]}..."
        )


# ---------------------------------------------------------------------------
# L1 contrastive-pair sampler (TODO 1)
# ---------------------------------------------------------------------------

def _sample_l1_contrastive(
    l1_pool: list[dict[str, Any]],
    *,
    n_pairs: int = 3,
    n_singles: int = 2,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample the L1 training block per the Distractor Invariance rule.

    Returns ``n_pairs * 2 + n_singles`` rows drawn from ``n_pairs + n_singles``
    distinct gold sequences. The pair rows share the same gold sequence but
    come from different distractor variants, so the model is forced to
    generalize across non-target zone noise.

    Raises if ``l1_pool`` does not contain enough unique golds or not enough
    distractor variants for the requested pairs.
    """
    l1_by_gold = _group_by_gold(l1_pool)
    # Need at least 2 variants per chosen "pair" gold.
    pair_eligible = [g for g, variants in l1_by_gold.items() if len(variants) >= 2]
    total_needed = n_pairs + n_singles
    if len(l1_by_gold) < total_needed:
        raise ValueError(
            f"L1 contrastive sampler: need {total_needed} unique golds but "
            f"only {len(l1_by_gold)} are available in the provided pool."
        )
    if len(pair_eligible) < n_pairs:
        raise ValueError(
            f"L1 contrastive sampler: need {n_pairs} golds with ≥2 distractor "
            f"variants each, but only {len(pair_eligible)} qualify."
        )

    # Prefer structural-class diversity: try to spread the 5 chosen golds
    # across distinct target_zone × {chain,ring} classes so the pairs
    # exercise multiple primitive structures.
    gold_to_rep_row = {g: l1_by_gold[g][0] for g in l1_by_gold}
    gold_to_class = {g: build_l1_structural_key(rep) for g, rep in gold_to_rep_row.items()}

    def _pick_spread(candidates: list[str], n: int) -> list[str]:
        by_class: dict[str, list[str]] = {}
        for g in candidates:
            by_class.setdefault(gold_to_class[g], []).append(g)
        for bucket in by_class.values():
            rng.shuffle(bucket)
        picked: list[str] = []
        # Round-robin across classes for maximum structural spread.
        while len(picked) < n and by_class:
            empties = []
            for cls, bucket in by_class.items():
                if not bucket:
                    empties.append(cls)
                    continue
                picked.append(bucket.pop())
                if len(picked) == n:
                    break
            for cls in empties:
                by_class.pop(cls, None)
        return picked

    shuffled_pair_pool = list(pair_eligible)
    rng.shuffle(shuffled_pair_pool)
    pair_golds = _pick_spread(shuffled_pair_pool, n_pairs)
    single_candidates = [g for g in l1_by_gold if g not in pair_golds]
    rng.shuffle(single_candidates)
    single_golds = _pick_spread(single_candidates, n_singles)
    if len(single_golds) < n_singles:
        # Fall back to arbitrary picks if structural spread was too restrictive.
        leftover = [g for g in single_candidates if g not in single_golds]
        rng.shuffle(leftover)
        single_golds.extend(leftover[: n_singles - len(single_golds)])

    sampled: list[dict[str, Any]] = []
    for g in pair_golds:
        variants = list(l1_by_gold[g])
        rng.shuffle(variants)
        sampled.extend(variants[:2])  # contrastive pair (2 distractor variants)
    for g in single_golds:
        variants = list(l1_by_gold[g])
        rng.shuffle(variants)
        sampled.append(variants[0])

    # Post-condition check: correct counts and gold multiplicities.
    assert len(sampled) == n_pairs * 2 + n_singles, (
        f"L1 contrastive sampler produced {len(sampled)} rows, "
        f"expected {n_pairs * 2 + n_singles}."
    )
    assert _unique_gold_count(sampled) == n_pairs + n_singles, (
        f"L1 contrastive sampler produced {_unique_gold_count(sampled)} unique "
        f"golds, expected {n_pairs + n_singles}."
    )
    rng.shuffle(sampled)
    return sampled


def _sample_l1_probe_exhaustive(
    l1_pool: list[dict[str, Any]],
    *,
    rng: random.Random,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Sample exactly one mission per unique L1 gold sequence for exhaustive probe.

    L1 has exactly 8 core topologies (4 target zones × 2 primitive branches:
    chain / ring). This returns 8 missions, one from each topology, providing
    a complete evaluation of L1 compositional logic.

    ``exclude_ids`` contains mission_ids already consumed by the L1 train
    sample — the probe must use a *different* distractor variant for any
    gold sequence that overlaps with train.
    """
    if exclude_ids is None:
        exclude_ids = set()

    groups = _group_by_gold(l1_pool)
    if len(groups) < 8:
        raise ValueError(
            f"L1 exhaustive probe: need 8 unique golds but pool has "
            f"only {len(groups)}."
        )

    sampled: list[dict[str, Any]] = []
    for gold_key in sorted(groups.keys()):
        # Prefer variants NOT already used in train.
        available = [r for r in groups[gold_key] if r["mission_id"] not in exclude_ids]
        if not available:
            # Fallback: all variants are used in train (should be very rare).
            available = list(groups[gold_key])
        rng.shuffle(available)
        sampled.append(available[0])

    # If the pool had > 8 unique golds, trim to exactly 8.
    if len(sampled) > 8:
        rng.shuffle(sampled)
        sampled = sampled[:8]

    assert len(sampled) == 8, (
        f"L1 exhaustive probe produced {len(sampled)} missions, expected 8."
    )
    rng.shuffle(sampled)
    return sampled


def _sample_l1_probe_holdout(
    l1_holdout_pool: list[dict[str, Any]],
    *,
    rng: random.Random,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Sample one mission per unique L1 gold sequence from a holdout-filtered pool.

    Used by Task 4 (holdout-small) to ensure ALL L1 probe missions satisfy ≥ 1
    holdout criterion. Because ``geosphere_chain`` (target=geosphere,
    condition=low_resource) never triggers either holdout predicate, the pool
    yields 7 unique golds rather than 8. Returns exactly one mission per gold,
    preferring distractor variants not already used in train.
    """
    if exclude_ids is None:
        exclude_ids = set()

    groups = _group_by_gold(l1_holdout_pool)
    if len(groups) == 0:
        raise ValueError("L1 holdout probe: holdout pool is empty.")

    sampled: list[dict[str, Any]] = []
    for gold_key in sorted(groups.keys()):
        available = [r for r in groups[gold_key] if r["mission_id"] not in exclude_ids]
        if not available:
            available = list(groups[gold_key])
        rng.shuffle(available)
        sampled.append(available[0])

    rng.shuffle(sampled)
    return sampled


def _split_disjoint_by_gold(
    pool: list[dict[str, Any]],
    *,
    train_n: int,
    probe_n: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a pool so train and probe receive DISJOINT gold keys.

    Picks ``train_n`` unique gold keys for train and ``probe_n`` unique gold
    keys for probe, with one mission drawn per gold. Raises if the pool has
    fewer than ``train_n + probe_n`` unique golds.

    Returns (train_rows, probe_rows, unused_rows).
    """
    groups = _group_by_gold(pool)
    gold_keys = sorted(groups.keys())
    if len(gold_keys) < train_n + probe_n:
        raise ValueError(
            f"_split_disjoint_by_gold: pool has {len(gold_keys)} unique golds, "
            f"need {train_n + probe_n} for a disjoint {train_n}/{probe_n} split."
        )
    rng.shuffle(gold_keys)
    train_keys = gold_keys[:train_n]
    probe_keys = gold_keys[train_n : train_n + probe_n]
    leftover_keys = gold_keys[train_n + probe_n :]

    train_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    unused: list[dict[str, Any]] = []
    for key in train_keys:
        variants = list(groups[key])
        rng.shuffle(variants)
        train_rows.append(variants[0])
        unused.extend(variants[1:])
    for key in probe_keys:
        variants = list(groups[key])
        rng.shuffle(variants)
        probe_rows.append(variants[0])
        unused.extend(variants[1:])
    for key in leftover_keys:
        unused.extend(groups[key])
    return train_rows, probe_rows, unused


def _split_scarcity_balanced(
    pool: list[dict[str, Any]],
    *,
    train_n: int,
    probe_n: int,
    train_affected: int,
    probe_affected: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Scarcity-balanced version of _split_disjoint_by_gold (for T6).

    Guarantees:
      - train and probe gold keys are disjoint (Global Uniqueness).
      - Exactly ``train_affected`` scarcity-affected golds land in train.
      - Exactly ``probe_affected`` scarcity-affected golds land in probe.
      - Train / probe totals are padded out with unaffected golds, also disjoint.
    """
    affected_pool = [r for r in pool if r.get("_scarcity_affected", False)]
    unaffected_pool = [r for r in pool if not r.get("_scarcity_affected", False)]

    train_unaffected = train_n - train_affected
    probe_unaffected = probe_n - probe_affected

    aff_train, aff_probe, aff_unused = _split_disjoint_by_gold(
        affected_pool, train_n=train_affected, probe_n=probe_affected, rng=rng,
    )
    un_train, un_probe, un_unused = _split_disjoint_by_gold(
        unaffected_pool, train_n=train_unaffected, probe_n=probe_unaffected, rng=rng,
    )

    train_rows = aff_train + un_train
    probe_rows = aff_probe + un_probe
    rng.shuffle(train_rows)
    rng.shuffle(probe_rows)
    return train_rows, probe_rows, aff_unused + un_unused


# ---------------------------------------------------------------------------
# Canonical split reader (reads tN_split columns from CSV)
# ---------------------------------------------------------------------------

def _allocate_from_csv(
    df: pd.DataFrame,
    env_runtime: dict[str, Any],
    *,
    task_number: int,
    split_option: str,
) -> "TaskSplit":
    """Read a canonical split from the ``tN_split`` column in the CSV."""
    col = f"t{task_number}_split"
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found in DataFrame. "
            f"Run scripts/generate_canonical_splits.py to add canonical split columns."
        )

    rows = df.to_dict(orient="records")
    for row in rows:
        row["_scarcity_affected"] = tag_scarcity_sensitivity(row, env_runtime)
        row["_goal_level"] = int(row["goal_level"])

    train_rows = [r for r in rows if str(r[col]) == "train"]
    probe_rows = [r for r in rows if str(r[col]) == "probe"]
    unused_rows = [r for r in rows if str(r[col]) not in ("train", "probe")]

    # Sort by presentation order column if available (eliminates curriculum bias)
    order_col = f"t{task_number}_order"
    if order_col in df.columns:
        def _order_key(r: dict) -> int:
            v = r.get(order_col)
            if pd.isna(v):
                return 999999
            return int(v)
        train_rows.sort(key=_order_key)
        probe_rows.sort(key=_order_key)

    summary = {
        "split_option": f"canonical-t{task_number}",
        "base_split": split_option,
        "holdout_criteria": [],
        "train_total": len(train_rows),
        "train_by_level": _level_counts(train_rows),
        "train_scarcity_affected": _scarcity_count(train_rows),
        "train_unique_golds": _unique_gold_count(train_rows),
        "probe_total": len(probe_rows),
        "probe_by_level": _level_counts(probe_rows),
        "probe_scarcity_affected": _scarcity_count(probe_rows),
        "probe_unique_golds": _unique_gold_count(probe_rows),
        "unused_total": len(unused_rows),
    }

    return TaskSplit(
        train_rows=train_rows,
        probe_rows=probe_rows,
        unused_rows=unused_rows,
        split_option=f"canonical-t{task_number}",
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Main split dispatcher
# ---------------------------------------------------------------------------

def allocate_split(
    df: pd.DataFrame,
    env_runtime: dict[str, Any],
    *,
    split_option: str,
    seed: int = 42,
    use_canonical: bool = False,
    task_number: int | None = None,
) -> TaskSplit:
    """Allocate the train/probe split for the named split option.

    Each row returned is tagged with ``_scarcity_affected`` and ``_goal_level``.

    Args:
        df: Full dataset (``dataset_state_complete.csv``).
        env_runtime: Compiled environment runtime.
        split_option: One of ``random-small``, ``goalwise-small``,
            ``holdout-small``, ``random-large``, ``transfer-probe``.
        seed: RNG seed for reproducible sampling (ignored when use_canonical=True).
        use_canonical: If True, read the split from ``tN_split`` CSV columns
            instead of computing it at runtime from the seed.
        task_number: Required when use_canonical=True (1-10).
    """
    if use_canonical:
        if task_number is None:
            raise ValueError("task_number is required when use_canonical=True")
        return _allocate_from_csv(df, env_runtime, task_number=task_number,
                                  split_option=split_option)

    if split_option not in SPLIT_OPTION_SPECS:
        raise ValueError(
            f"split_option must be one of {sorted(SPLIT_OPTION_SPECS.keys())}"
        )

    base_split, holdout_criteria = SPLIT_OPTION_SPECS[split_option]

    rng = random.Random(seed)
    rows = df.to_dict(orient="records")

    # Tag every row before splitting.
    for row in rows:
        row["_scarcity_affected"] = tag_scarcity_sensitivity(row, env_runtime)
        row["_goal_level"] = int(row["goal_level"])

    l1 = [r for r in rows if r["_goal_level"] == 1]
    l2 = [r for r in rows if r["_goal_level"] == 2]
    l3 = [r for r in rows if r["_goal_level"] == 3]
    l4 = [r for r in rows if r["_goal_level"] == 4]

    if base_split == "random_small":
        result = _allocate_random_small(l1, l2, l3, rng)
    elif base_split == "accelerated_small":
        result = _allocate_accelerated_small(l1, l2, l3, rng)
    elif base_split == "goalwise_small":
        result = _allocate_goalwise_small(l1, l2, l3, rng)
    elif base_split == "holdout_small":
        result = _allocate_holdout_small(l1, l2, l3, holdout_criteria, rng)
    elif base_split == "random_large":
        result = _allocate_random_large(l1, l2, l3, rng)
    elif base_split == "transfer_probe":
        result = _allocate_transfer_probe(l1, l2, l3, l4, rng)
    else:
        raise ValueError(f"Unknown base_split '{base_split}' for '{split_option}'.")

    train_rows, probe_rows, unused = result
    _verify_disjoint(train_rows, probe_rows)

    rng.shuffle(train_rows)
    rng.shuffle(probe_rows)

    summary = {
        "split_option": split_option,
        "base_split": base_split,
        "holdout_criteria": holdout_criteria,
        "train_total": len(train_rows),
        "train_by_level": _level_counts(train_rows),
        "train_scarcity_affected": _scarcity_count(train_rows),
        "train_unique_golds": _unique_gold_count(train_rows),
        "probe_total": len(probe_rows),
        "probe_by_level": _level_counts(probe_rows),
        "probe_scarcity_affected": _scarcity_count(probe_rows),
        "probe_unique_golds": _unique_gold_count(probe_rows),
        "unused_total": len(unused),
    }

    return TaskSplit(
        train_rows=train_rows,
        probe_rows=probe_rows,
        unused_rows=unused,
        split_option=split_option,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# random-small — Tasks 1, 2, 5 — (24/24)
# L1 (8/8)  L2 (8/8)  L3 (8/8)
# L1 train = 3 contrastive pairs + 2 singles (8 missions, 5 unique golds)
# L1 probe = exhaustive 8 topologies (8 missions, 8 unique golds)
# ---------------------------------------------------------------------------

def _allocate_random_small(
    l1: list[dict], l2: list[dict], l3: list[dict], rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    l1_train = _sample_l1_contrastive(l1, n_pairs=3, n_singles=2, rng=rng)
    l1_train_ids = {r["mission_id"] for r in l1_train}
    l1_probe = _sample_l1_probe_exhaustive(l1, rng=rng, exclude_ids=l1_train_ids)

    l2_train, l2_probe, l2_unused = _split_disjoint_by_gold(
        l2, train_n=8, probe_n=8, rng=rng,
    )
    l3_train, l3_probe, l3_unused = _split_disjoint_by_gold(
        l3, train_n=8, probe_n=8, rng=rng,
    )

    l1_selected_ids = {r["mission_id"] for r in l1_train + l1_probe}
    l1_unused = [r for r in l1 if r["mission_id"] not in l1_selected_ids]

    return (
        l1_train + l2_train + l3_train,
        l1_probe + l2_probe + l3_probe,
        l1_unused + l2_unused + l3_unused,
    )


# ---------------------------------------------------------------------------
# accelerated-small — Task 3 — (12/24)
# L1 (3/8)  L2 (4/8)  L3 (5/8)
# Half the training exposure of random-small. L1 train has 3 unique golds
# from distinct structural classes (no contrastive pairs — insufficient budget).
# Same 24-probe pool structure as random-small.
# ---------------------------------------------------------------------------

def _allocate_accelerated_small(
    l1: list[dict], l2: list[dict], l3: list[dict], rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    # L1 train: 3 missions with unique golds from distinct structural classes.
    # No contrastive pairs (budget too small for pairs).
    l1_by_gold = _group_by_gold(l1)
    gold_to_rep = {g: rows[0] for g, rows in l1_by_gold.items()}
    gold_to_class = {g: build_l1_structural_key(rep) for g, rep in gold_to_rep.items()}

    # Pick 3 golds spread across structural classes.
    by_class: dict[str, list[str]] = {}
    for g, cls in gold_to_class.items():
        by_class.setdefault(cls, []).append(g)
    for bucket in by_class.values():
        rng.shuffle(bucket)

    picked_golds: list[str] = []
    while len(picked_golds) < 3 and by_class:
        empties = []
        for cls, bucket in sorted(by_class.items()):
            if not bucket:
                empties.append(cls)
                continue
            picked_golds.append(bucket.pop())
            if len(picked_golds) == 3:
                break
        for cls in empties:
            by_class.pop(cls, None)

    l1_train: list[dict] = []
    for g in picked_golds:
        variants = list(l1_by_gold[g])
        rng.shuffle(variants)
        l1_train.append(variants[0])
    rng.shuffle(l1_train)

    l1_train_ids = {r["mission_id"] for r in l1_train}
    l1_probe = _sample_l1_probe_exhaustive(l1, rng=rng, exclude_ids=l1_train_ids)

    # L2: 4 train / 8 probe, globally unique golds.
    l2_train, l2_probe, l2_unused = _split_disjoint_by_gold(
        l2, train_n=4, probe_n=8, rng=rng,
    )
    # L3: 5 train / 8 probe, globally unique golds.
    l3_train, l3_probe, l3_unused = _split_disjoint_by_gold(
        l3, train_n=5, probe_n=8, rng=rng,
    )

    l1_selected_ids = {r["mission_id"] for r in l1_train + l1_probe}
    l1_unused = [r for r in l1 if r["mission_id"] not in l1_selected_ids]

    return (
        l1_train + l2_train + l3_train,
        l1_probe + l2_probe + l3_probe,
        l1_unused + l2_unused + l3_unused,
    )


# ---------------------------------------------------------------------------
# goalwise-small — Task 4 — (24/24)
# L1 (8/0)  L2 (16/0)  L3 (0/24)
# Train is L1+L2 only; probe is exclusively L3 (depth extrapolation).
# ---------------------------------------------------------------------------

def _allocate_goalwise_small(
    l1: list[dict], l2: list[dict], l3: list[dict], rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    l1_train = _sample_l1_contrastive(l1, n_pairs=3, n_singles=2, rng=rng)

    # L2: all 16 unique golds go to train.
    l2_train, _empty_probe, l2_unused = _split_disjoint_by_gold(
        l2, train_n=16, probe_n=0, rng=rng,
    )

    # L3: 24 unique golds go to probe (L3 has 32 unique golds → take 24).
    _empty_train, l3_probe, l3_unused = _split_disjoint_by_gold(
        l3, train_n=0, probe_n=24, rng=rng,
    )

    l1_selected_ids = {r["mission_id"] for r in l1_train}
    l1_unused = [r for r in l1 if r["mission_id"] not in l1_selected_ids]

    return (
        l1_train + l2_train,
        l3_probe,
        l1_unused + l2_unused + l3_unused,
    )


# ---------------------------------------------------------------------------
# holdout-small — Task 4 — (24/24)
# L1 (8/7)  L2 (8/8)  L3 (8/9)
# Train rows satisfy NO holdout criterion; probe rows satisfy ≥ 1.
# L1 probe restricted to holdout pool → 7 unique topologies (geosphere_chain
# has no holdout variant). L3 probe bumped to 9 to keep total probe at 24.
# (L2 is capped at 8+8=16 unique golds; L3 has 32 with no shared golds.)
# ---------------------------------------------------------------------------

def _holdout_gold_split(
    pool: list[dict[str, Any]],
    holdout_criteria: list[str],
    *,
    train_n: int,
    probe_n: int,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Allocate gold-disjoint train / probe quotas from a holdout-partitioned pool.

    Because holdout predicates are row-level, the same gold sequence can
    legitimately appear in both holdout and non-holdout rows (when the
    holdout-triggering zone is not on the target's dependency chain — e.g.,
    target=reservoir with geosphere=bad_quality vs geosphere=ideal).

    Algorithm:
      1. Partition rows into holdout vs non-holdout.
      2. Compute only-holdout, only-non-holdout, and shared gold keys.
      3. Fill probe first from only-holdout golds, then from shared.
      4. Fill train first from only-non-holdout golds, then from any
         remaining shared golds. This ordering guarantees we never waste a
         shared gold on one side when the other side needs it.
    """
    holdout_rows = [r for r in pool if _matches_any_holdout(r, holdout_criteria)]
    nonholdout_rows = [r for r in pool if not _matches_any_holdout(r, holdout_criteria)]

    holdout_golds = {str(r["gold_action_sequence"]) for r in holdout_rows}
    non_golds = {str(r["gold_action_sequence"]) for r in nonholdout_rows}
    only_holdout = sorted(holdout_golds - non_golds)
    only_non = sorted(non_golds - holdout_golds)
    shared = sorted(holdout_golds & non_golds)
    rng.shuffle(only_holdout)
    rng.shuffle(only_non)
    rng.shuffle(shared)

    # Probe quota — prefer only-holdout, then borrow from shared.
    probe_golds_list: list[str] = list(only_holdout[:probe_n])
    shortfall = probe_n - len(probe_golds_list)
    if shortfall > 0:
        take = shared[:shortfall]
        probe_golds_list.extend(take)
        shared = shared[shortfall:]
    if len(probe_golds_list) < probe_n:
        raise ValueError(
            f"holdout-small probe: only {len(probe_golds_list)} unique golds "
            f"available from holdout pool (need {probe_n}). "
            f"only_holdout={len(only_holdout)}, shared_initial={len(shared) + shortfall}"
        )
    probe_golds = set(probe_golds_list)

    # Train quota — prefer only-non-holdout, then use any shared golds not
    # already consumed by probe.
    train_golds_list: list[str] = list(only_non[:train_n])
    shortfall = train_n - len(train_golds_list)
    if shortfall > 0:
        take = shared[:shortfall]
        train_golds_list.extend(take)
    if len(train_golds_list) < train_n:
        raise ValueError(
            f"holdout-small train: only {len(train_golds_list)} unique golds "
            f"available from non-holdout pool (need {train_n}). "
            f"only_non={len(only_non)}, shared_remaining={len(shared)}"
        )
    train_golds = set(train_golds_list)

    assert not (train_golds & probe_golds), (
        "holdout-small: train and probe gold sets overlap — allocator bug."
    )

    # Materialize rows from the selected gold keys.
    probe_row_pool = [r for r in holdout_rows if str(r["gold_action_sequence"]) in probe_golds]
    train_row_pool = [r for r in nonholdout_rows if str(r["gold_action_sequence"]) in train_golds]

    train_rows, _, train_unused = _split_disjoint_by_gold(
        train_row_pool, train_n=train_n, probe_n=0, rng=rng,
    )
    _, probe_rows, probe_unused = _split_disjoint_by_gold(
        probe_row_pool, train_n=0, probe_n=probe_n, rng=rng,
    )
    return train_rows, probe_rows, train_unused + probe_unused


def _allocate_holdout_small(
    l1: list[dict], l2: list[dict], l3: list[dict],
    holdout_criteria: list[str], rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    # L1 train: contrastive sample drawn from non-holdout L1 rows.
    l1_train_pool = [r for r in l1 if not _matches_any_holdout(r, holdout_criteria)]
    l1_train = _sample_l1_contrastive(l1_train_pool, n_pairs=3, n_singles=2, rng=rng)
    l1_train_ids = {r["mission_id"] for r in l1_train}

    # L1 probe: all unique golds from the holdout-filtered L1 pool.
    # geosphere_chain (target=geosphere, condition=low_resource) has no holdout
    # variant, so this yields 7 unique topologies, not 8. Every probe mission
    # satisfies ≥ 1 holdout criterion (same policy as L2/L3 probe).
    l1_holdout_pool = [r for r in l1 if _matches_any_holdout(r, holdout_criteria)]
    l1_probe = _sample_l1_probe_holdout(l1_holdout_pool, rng=rng, exclude_ids=l1_train_ids)
    # len(l1_probe) == 7; L3 probe is bumped to 9 to keep total probe = 24.

    l2_train, l2_probe, _ = _holdout_gold_split(
        l2, holdout_criteria, train_n=8, probe_n=8, rng=rng,
    )
    # L3 probe = 9 (not 8) to compensate for L1 probe dropping from 8 → 7.
    # L3 has 32 unique golds with clean holdout/non-holdout separation.
    l3_train, l3_probe, _ = _holdout_gold_split(
        l3, holdout_criteria, train_n=8, probe_n=9, rng=rng,
    )

    selected_ids = {r["mission_id"] for r in
                    l1_train + l1_probe + l2_train + l2_probe + l3_train + l3_probe}
    unused = [r for r in l1 + l2 + l3 if r["mission_id"] not in selected_ids]

    return (
        l1_train + l2_train + l3_train,
        l1_probe + l2_probe + l3_probe,
        unused,
    )


# ---------------------------------------------------------------------------
# random-large — Tasks 6, 7 — (32/24)
# L1 (8/0)  L2 (8/8)  L3 (16/16)
# For T6: scarcity-affected gold keys are split 2/2 (L2) and 8/8 (L3)
#         between train and probe. This also works for T7 because the
#         integrate rule affects every mission equally.
# ---------------------------------------------------------------------------

def _allocate_random_large(
    l1: list[dict], l2: list[dict], l3: list[dict], rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    l1_train = _sample_l1_contrastive(l1, n_pairs=3, n_singles=2, rng=rng)

    # L2: 8/8 split with 2/2 of the scarcity-affected golds on each side.
    l2_train, l2_probe, l2_unused = _split_scarcity_balanced(
        l2, train_n=8, probe_n=8, train_affected=2, probe_affected=2, rng=rng,
    )

    # L3: 16/16 split with 8/8 of the scarcity-affected golds on each side.
    l3_train, l3_probe, l3_unused = _split_scarcity_balanced(
        l3, train_n=16, probe_n=16, train_affected=8, probe_affected=8, rng=rng,
    )

    l1_selected_ids = {r["mission_id"] for r in l1_train}
    l1_unused = [r for r in l1 if r["mission_id"] not in l1_selected_ids]

    return (
        l1_train + l2_train + l3_train,
        l2_probe + l3_probe,
        l1_unused + l2_unused + l3_unused,
    )


# ---------------------------------------------------------------------------
# transfer-probe — Task 8 EFGHI — (0/32)
# 0 train / 32 probe (8×L1 + 8×L2 + 8×L3 + 8×L4)
# Pure probe split for the transfer environment — no training on EFGHI.
# Globally unique golds across the probe set.
# L4 probes test depth-4 nesting (unique to 5-zone EFGHI).
# ---------------------------------------------------------------------------

def _allocate_transfer_probe(
    l1: list[dict], l2: list[dict], l3: list[dict],
    l4: list[dict], rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    _, l1_probe, l1_unused = _split_disjoint_by_gold(
        l1, train_n=0, probe_n=8, rng=rng,
    )
    _, l2_probe, l2_unused = _split_disjoint_by_gold(
        l2, train_n=0, probe_n=8, rng=rng,
    )
    _, l3_probe, l3_unused = _split_disjoint_by_gold(
        l3, train_n=0, probe_n=8, rng=rng,
    )
    if l4:
        _, l4_probe, l4_unused = _split_disjoint_by_gold(
            l4, train_n=0, probe_n=8, rng=rng,
        )
    else:
        l4_probe, l4_unused = [], []
    return (
        [],
        l1_probe + l2_probe + l3_probe + l4_probe,
        l1_unused + l2_unused + l3_unused + l4_unused,
    )


# ---------------------------------------------------------------------------
# Adaptation sub-pool allocation
# ---------------------------------------------------------------------------

def build_adaptation_pools(
    train_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    env_runtime: dict[str, Any],
    *,
    rule_type: str,
    seed: int = 42,
    zero_shot_count: int = 0,
    train_per_block: int = 2,
    probe_per_block: int = 4,
    num_blocks: int = 5,
    final_probe_count: int = 0,
) -> AdaptationPools:
    """Build adaptation sub-pools for T6 (scarcity) and T7 (integrate).

    Defaults now match the plan:
      - 5 interleaved Train/Probe blocks
      - Each Train: 2 missions (all rule-affected)
      - Each Probe: 4 missions (T6: 2 affected + 2 not-affected; T7: 4 affected)
      - No zero-shot prologue and no enlarged final probe.

    The adaptation pool is sampled FROM the main study train/probe splits,
    so disjointness to the Main Study is automatic.
    """
    rng = random.Random(seed)

    # Ensure scarcity tagging is present on every row.
    for r in train_rows + probe_rows:
        if "_scarcity_affected" not in r:
            r["_scarcity_affected"] = tag_scarcity_sensitivity(r, env_runtime)

    if rule_type == "scarcity":
        return _build_scarcity_pools(
            train_rows, probe_rows, rng,
            zero_shot_count, train_per_block, probe_per_block,
            num_blocks, final_probe_count,
        )
    elif rule_type == "integrate":
        return _build_integrate_pools(
            train_rows, probe_rows, rng,
            zero_shot_count, train_per_block, probe_per_block,
            num_blocks, final_probe_count,
        )
    else:
        raise ValueError(
            f"Unknown rule_type: {rule_type}. Must be 'scarcity' or 'integrate'."
        )


def _build_scarcity_pools(
    train_rows, probe_rows, rng,
    zero_shot_count, train_per_block, probe_per_block,
    num_blocks, final_probe_count,
) -> AdaptationPools:
    """T6 scarcity adaptation: balanced affected / not-affected probes.

    Plan §Task 6:
      - Train: 10 missions = 5 blocks × 2 scarcity-affected
      - Probe: 20 missions = 5 blocks × (2 affected + 2 not-affected)
    """
    train_affected = [r for r in train_rows if r.get("_scarcity_affected", False)]
    probe_affected = [r for r in probe_rows if r.get("_scarcity_affected", False)]
    probe_not_affected = [r for r in probe_rows if not r.get("_scarcity_affected", False)]

    rng.shuffle(train_affected)
    rng.shuffle(probe_affected)
    rng.shuffle(probe_not_affected)

    # Optional zero-shot probe (disabled by default).
    zs_half = zero_shot_count // 2
    zero_shot = probe_affected[:zs_half] + probe_not_affected[:zs_half]
    rng.shuffle(zero_shot)
    rem_aff = list(probe_affected[zs_half:])
    rem_naff = list(probe_not_affected[zs_half:])

    # Train blocks: all affected, train_per_block per block.
    total_train_needed = train_per_block * num_blocks
    adapt_train = train_affected[:total_train_needed]
    train_blocks: list[list[dict]] = [
        adapt_train[i * train_per_block : (i + 1) * train_per_block]
        for i in range(num_blocks)
    ]

    # Probe blocks: balanced affected / not-affected.
    probe_per_side = probe_per_block // 2
    probe_blocks: list[list[dict]] = []
    aff_idx = 0
    naff_idx = 0
    for _ in range(num_blocks):
        block = (
            rem_aff[aff_idx : aff_idx + probe_per_side]
            + rem_naff[naff_idx : naff_idx + probe_per_side]
        )
        rng.shuffle(block)
        probe_blocks.append(block)
        aff_idx += probe_per_side
        naff_idx += probe_per_side

    final_half = final_probe_count // 2
    final_probe = (
        rem_aff[aff_idx : aff_idx + final_half]
        + rem_naff[naff_idx : naff_idx + final_half]
    )
    rng.shuffle(final_probe)

    summary = {
        "rule_type": "scarcity",
        "zero_shot_count": len(zero_shot),
        "train_blocks": [len(b) for b in train_blocks],
        "probe_blocks": [len(b) for b in probe_blocks],
        "final_probe_count": len(final_probe),
        "total_adaptation_train": sum(len(b) for b in train_blocks),
        "total_adaptation_probe": sum(len(b) for b in probe_blocks) + len(final_probe),
    }

    return AdaptationPools(
        zero_shot_probe_rows=zero_shot,
        train_blocks=train_blocks,
        probe_blocks=probe_blocks,
        final_probe_rows=final_probe,
        summary=summary,
    )


def _build_integrate_pools(
    train_rows, probe_rows, rng,
    zero_shot_count, train_per_block, probe_per_block,
    num_blocks, final_probe_count,
) -> AdaptationPools:
    """T7 integrate adaptation: every mission is rule-affected.

    Plan §Task 7:
      - Train: 10 missions = 2 L2 + 8 L3 → 5 blocks × 2
      - Probe: 20 missions = 4 L2 + 16 L3 → 5 blocks × 4
    """
    # Split by goal level so we can honour the plan's (2 L2 + 8 L3) train
    # and (4 L2 + 16 L3) probe composition when possible.
    train_l2 = [r for r in train_rows if int(r.get("_goal_level", r.get("goal_level", 0))) == 2]
    train_l3 = [r for r in train_rows if int(r.get("_goal_level", r.get("goal_level", 0))) == 3]
    probe_l2 = [r for r in probe_rows if int(r.get("_goal_level", r.get("goal_level", 0))) == 2]
    probe_l3 = [r for r in probe_rows if int(r.get("_goal_level", r.get("goal_level", 0))) == 3]
    rng.shuffle(train_l2)
    rng.shuffle(train_l3)
    rng.shuffle(probe_l2)
    rng.shuffle(probe_l3)

    total_train = train_per_block * num_blocks     # 10
    total_probe = probe_per_block * num_blocks      # 20

    # Train target: prefer 2 L2 + 8 L3 when available; otherwise fill from whichever level has rows.
    target_train_l2 = min(2, len(train_l2), total_train)
    target_train_l3 = min(total_train - target_train_l2, len(train_l3))
    adapt_train = train_l2[:target_train_l2] + train_l3[:target_train_l3]
    # Pad from the other level if we are short.
    if len(adapt_train) < total_train:
        remainder_pool = train_l2[target_train_l2:] + train_l3[target_train_l3:]
        adapt_train.extend(remainder_pool[: total_train - len(adapt_train)])

    # Probe target: prefer 4 L2 + 16 L3 when available.
    target_probe_l2 = min(4, len(probe_l2), total_probe)
    target_probe_l3 = min(total_probe - target_probe_l2, len(probe_l3))
    adapt_probe = probe_l2[:target_probe_l2] + probe_l3[:target_probe_l3]
    if len(adapt_probe) < total_probe:
        remainder_probe = probe_l2[target_probe_l2:] + probe_l3[target_probe_l3:]
        adapt_probe.extend(remainder_probe[: total_probe - len(adapt_probe)])

    rng.shuffle(adapt_train)
    rng.shuffle(adapt_probe)

    # Optional zero-shot prologue (disabled by default).
    zero_shot = adapt_probe[:zero_shot_count]
    remaining_probe = adapt_probe[zero_shot_count:]

    train_blocks: list[list[dict]] = [
        adapt_train[i * train_per_block : (i + 1) * train_per_block]
        for i in range(num_blocks)
    ]

    probe_blocks: list[list[dict]] = []
    p_idx = 0
    for _ in range(num_blocks):
        probe_blocks.append(remaining_probe[p_idx : p_idx + probe_per_block])
        p_idx += probe_per_block

    final_probe = remaining_probe[p_idx : p_idx + final_probe_count]

    summary = {
        "rule_type": "integrate",
        "zero_shot_count": len(zero_shot),
        "train_blocks": [len(b) for b in train_blocks],
        "probe_blocks": [len(b) for b in probe_blocks],
        "final_probe_count": len(final_probe),
        "total_adaptation_train": sum(len(b) for b in train_blocks),
        "total_adaptation_probe": sum(len(b) for b in probe_blocks) + len(final_probe),
    }

    return AdaptationPools(
        zero_shot_probe_rows=zero_shot,
        train_blocks=train_blocks,
        probe_blocks=probe_blocks,
        final_probe_rows=final_probe,
        summary=summary,
    )
