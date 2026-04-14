from __future__ import annotations

import random
import re
from typing import Any

from .context_blocks import (
    CONTEXT_BLOCKS,
    EFGHI_BASELINE_PROMPT,
    EFGHI_COMPOSITION_SYSTEM_PROMPT,
    EFGHI_ENVIRONMENT_PROMPT,
    TRANSFER_TRANSITION_ANNOUNCEMENT,
)

# ---------------------------------------------------------------------------
# Context Initialization Options (spec §1)
# ---------------------------------------------------------------------------
# "a" = Baseline + Environment + Composition System  (full context)
# "b" = Baseline + Environment                       (no composition rules)
# "c" = Baseline only                                (minimal)

CONTEXT_INIT_OPTIONS: dict[str, list[str]] = {
    "a": ["baseline", "environment", "composition_system"],
    "b": ["baseline", "environment"],
    "c": ["baseline"],
    # Named aliases used by the 10-task design
    "full-context":         ["baseline", "environment", "composition_system"],
    "environment-context":  ["baseline", "environment"],
    "no-context":           ["baseline"],
    # EFGHI context options (for T8 transfer phase)
    "efghi-full-context":         ["efghi_baseline", "efghi_environment", "efghi_composition_system"],
    "efghi-environment-context":  ["efghi_baseline", "efghi_environment"],
}

CONTEXT_INIT_LABELS: dict[str, str] = {
    "a": "Baseline + Environment + Composition System",
    "b": "Baseline + Environment",
    "c": "Baseline only",
    # Named aliases
    "full-context":         "Baseline + Environment + Composition System",
    "environment-context":  "Baseline + Environment",
    "no-context":           "Baseline only",
    # EFGHI
    "efghi-full-context":         "EFGHI Baseline + Environment + Composition System",
    "efghi-environment-context":  "EFGHI Baseline + Environment",
}

# ---------------------------------------------------------------------------
# Prompt Style Configuration (spec §2.2)
# ---------------------------------------------------------------------------
# "a" = Train and test on only canonical
# "b" = Randomly assign a style for both training and testing
# "c" = Train canonical only; randomly assign non-canonical for test

PROMPT_STYLE_CONFIG_OPTIONS: dict[str, str] = {
    "a": "canonical_only",
    "b": "random_all",
    "c": "canonical_train_random_test",
    # Named aliases used by the new 7-task design
    "canonical-canonical":     "canonical_only",
    "canonical-non_canonical": "canonical_train_random_test",
}

ALL_PROMPT_STYLES = [
    "canonical",
    "zone_reordered",
    "word_resource_absent",
    "prose_like",
    "zone_name_absent",
]

NON_CANONICAL_STYLES = [s for s in ALL_PROMPT_STYLES if s != "canonical"]

# ---------------------------------------------------------------------------
# Zone-name and resource mappings for prompt styles
# ---------------------------------------------------------------------------

ZONE_ORDER_CANONICAL = ["atmosphere", "biomass", "geosphere", "reservoir"]
ZONE_ORDER_REORDERED = ["geosphere", "biomass", "atmosphere", "reservoir"]

ZONE_RESOURCE_NAMES = {
    # zone-name-absent uses the exact terminal resource tokens per the prompt
    # spec so generalization isn't confounded by lexical hints (§3 of
    # new_implementations_todo.md).
    "atmosphere": "air",
    "biomass": "plant",
    "geosphere": "mineral",
    "reservoir": "water",
}


# ---------------------------------------------------------------------------
# Condition formatting per prompt style
# ---------------------------------------------------------------------------

def _parse_k_level(condition_str: str) -> int | None:
    """Extract k from 'bad_quality_<k>' or return None."""
    if condition_str.startswith("bad_quality_"):
        return int(condition_str.split("_")[-1])
    return None


def format_condition(condition_str: str, prompt_style: str) -> str:
    """Map internal condition string to the style-specific wording from the spec."""
    k = _parse_k_level(condition_str)

    if prompt_style in ("canonical", "zone_reordered"):
        if condition_str == "ideal":
            return "ideal resource"
        if condition_str == "low_resource":
            return "low on resource"
        if k is not None:
            return f"bad quality of resource (severity level = {k})"

    elif prompt_style == "word_resource_absent":
        if condition_str == "ideal":
            return "excellent"
        if condition_str == "low_resource":
            return "under supplied"
        if k is not None:
            return f"{k}-degree quality fault"

    elif prompt_style == "prose_like":
        if condition_str == "ideal":
            return "looking ideal"
        if condition_str == "low_resource":
            return "currently low on resource"
        if k is not None:
            return f"suffering from a {k}-degree resource contamination"

    elif prompt_style == "zone_name_absent":
        if condition_str == "ideal":
            return "Ideal"
        if condition_str == "low_resource":
            return "Low on"
        if k is not None:
            return f"{k}-level-contaminated"

    raise ValueError(f"Cannot format condition '{condition_str}' for style '{prompt_style}'")


# ---------------------------------------------------------------------------
# Instruction block assembly
# ---------------------------------------------------------------------------

def build_instruction_block(
    *,
    context_init_option: str,
    include_output_format_prompt: bool = True,
) -> str:
    """Assemble the static instruction block from the selected context init blocks."""
    if context_init_option not in CONTEXT_INIT_OPTIONS:
        valid = ", ".join(sorted(CONTEXT_INIT_OPTIONS.keys()))
        raise ValueError(
            f"Unsupported context_init_option '{context_init_option}'. Valid: {valid}"
        )
    block_keys = CONTEXT_INIT_OPTIONS[context_init_option]
    blocks = [CONTEXT_BLOCKS[key] for key in block_keys]

    if include_output_format_prompt:
        blocks.append(
            "Output format:\n"
            'Return ONLY a single raw JSON object with key "actions" '
            "and a non-empty list of action tokens.\n"
            'Example: {"actions": ["AIR", "PLANT", "WATER"]}'
        )

    return "\n\n".join(blocks).strip()


# ---------------------------------------------------------------------------
# Mission prompt assembly — 5 styles
# ---------------------------------------------------------------------------

def _build_canonical_mission(row: dict[str, Any]) -> str:
    target = row["target_zone"]
    lines = ["Current state of the ecosystem:"]
    for zone in ZONE_ORDER_CANONICAL:
        cond = format_condition(row[f"{zone}_condition"], "canonical")
        lines.append(f"{zone}: {cond}")
    lines.append(f"Manage {target} by doing as few actions as possible. What is your complete action sequence?")
    return "\n".join(lines)


def _build_zone_reordered_mission(row: dict[str, Any]) -> str:
    target = row["target_zone"]
    lines = ["Current state of the ecosystem:"]
    for zone in ZONE_ORDER_REORDERED:
        cond = format_condition(row[f"{zone}_condition"], "zone_reordered")
        lines.append(f"{zone}: {cond}")
    lines.append(f"Manage {target} by doing as few actions as possible. What is your complete action sequence?")
    return "\n".join(lines)


def _build_word_resource_absent_mission(row: dict[str, Any]) -> str:
    target = row["target_zone"]
    lines = ["Current state of the ecosystem:"]
    for zone in ZONE_ORDER_CANONICAL:
        cond = format_condition(row[f"{zone}_condition"], "word_resource_absent")
        lines.append(f"{zone}: {cond}")
    lines.append(f"Manage {target} by doing as few actions as possible. What is your complete action sequence?")
    return "\n".join(lines)


def _build_prose_like_mission(row: dict[str, Any]) -> str:
    target = row["target_zone"]
    atm = format_condition(row["atmosphere_condition"], "prose_like")
    bio = format_condition(row["biomass_condition"], "prose_like")
    geo = format_condition(row["geosphere_condition"], "prose_like")
    res = format_condition(row["reservoir_condition"], "prose_like")
    return (
        f"You need to manage {target} with as few actions as you can. "
        f"The atmosphere is {atm} and the biomass is {bio}. "
        f"The geosphere is {geo} while the reservoir is {res}. "
        "What is your complete action sequence?"
    )


def _build_zone_name_absent_mission(row: dict[str, Any]) -> str:
    target = row["target_zone"]
    lines = ["Current state of the resources in the ecosystem:"]
    for zone in ZONE_ORDER_CANONICAL:
        cond = format_condition(row[f"{zone}_condition"], "zone_name_absent")
        resource = ZONE_RESOURCE_NAMES[zone]
        lines.append(f"{cond} {resource}")
    lines.append(f"Manage {target} by doing as few actions as possible. What is your complete action sequence?")
    return "\n".join(lines)


_MISSION_BUILDERS = {
    "canonical": _build_canonical_mission,
    "zone_reordered": _build_zone_reordered_mission,
    "word_resource_absent": _build_word_resource_absent_mission,
    "prose_like": _build_prose_like_mission,
    "zone_name_absent": _build_zone_name_absent_mission,
}


def build_mission_text(
    *,
    row: dict[str, Any],
    prompt_style: str,
) -> str:
    """Build *only* the mission-specific prompt (no instruction block).

    Use this when the instruction block has already been sent as a session
    prologue, so each subsequent mission message is short and context-efficient.
    """
    if prompt_style not in _MISSION_BUILDERS:
        valid = ", ".join(sorted(_MISSION_BUILDERS.keys()))
        raise ValueError(f"Unsupported prompt_style '{prompt_style}'. Valid: {valid}")
    return _MISSION_BUILDERS[prompt_style](row)


def build_mission_prompt(
    *,
    row: dict[str, Any],
    prompt_style: str,
    instruction_block: str,
) -> str:
    """Combine instruction block + mission-specific prompt.

    .. deprecated::
        Prefer sending the instruction_block once as a session prologue
        and using :func:`build_mission_text` for each mission.
    """
    mission_text = build_mission_text(row=row, prompt_style=prompt_style)
    return instruction_block + "\n\n" + mission_text


# ---------------------------------------------------------------------------
# Prompt style resolution
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# EFGHI mission prompt builder
# ---------------------------------------------------------------------------

EFGHI_ZONE_ORDER = ["furnace", "pipeline", "conduit", "archive", "basin"]


def _build_efghi_canonical_mission(row: dict[str, Any]) -> str:
    target = row["target_zone"]
    lines = ["Current state of the facility:"]
    for zone in EFGHI_ZONE_ORDER:
        cond = format_condition(row[f"{zone}_condition"], "canonical")
        lines.append(f"{zone}: {cond}")
    lines.append(f"Manage {target} by doing as few actions as possible. What is your complete action sequence?")
    return "\n".join(lines)


def build_efghi_mission_text(*, row: dict[str, Any]) -> str:
    """Build mission prompt for EFGHI environment (canonical format only)."""
    return _build_efghi_canonical_mission(row)


# ---------------------------------------------------------------------------
# Transfer transition prompt builder (T8)
# ---------------------------------------------------------------------------

def build_transfer_transition_prompt(condition: str) -> str:
    """Build the transfer transition prompt for T8 per condition.

    All conditions receive: Transition Announcement + EFGHI Baseline + EFGHI Environment.
    Ceiling additionally receives: EFGHI Composition System.
    """
    blocks = [
        TRANSFER_TRANSITION_ANNOUNCEMENT,
        EFGHI_BASELINE_PROMPT,
        EFGHI_ENVIRONMENT_PROMPT,
    ]
    if condition == "ceiling":
        blocks.append(EFGHI_COMPOSITION_SYSTEM_PROMPT)
    # Also add the output format prompt
    blocks.append(
        "Output format:\n"
        'Return ONLY a single raw JSON object with key "actions" '
        "and a non-empty list of action tokens.\n"
        'Example: {"actions": ["HEAT", "FLOW", "WIRE"]}'
    )
    return "\n\n".join(blocks).strip()


# ---------------------------------------------------------------------------
# Prompt style resolution
# ---------------------------------------------------------------------------

def resolve_prompt_style(
    prompt_style_config: str,
    is_train: bool,
    rng: random.Random,
) -> str:
    """Select prompt style for a mission based on config option."""
    if prompt_style_config not in PROMPT_STYLE_CONFIG_OPTIONS:
        valid = ", ".join(sorted(PROMPT_STYLE_CONFIG_OPTIONS.keys()))
        raise ValueError(
            f"Unsupported prompt_style_config '{prompt_style_config}'. Valid: {valid}"
        )
    mode = PROMPT_STYLE_CONFIG_OPTIONS[prompt_style_config]
    if mode == "canonical_only":
        return "canonical"
    if mode == "random_all":
        return rng.choice(ALL_PROMPT_STYLES)
    # canonical_train_random_test
    if is_train:
        return "canonical"
    return rng.choice(NON_CANONICAL_STYLES)
