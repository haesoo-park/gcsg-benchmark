from __future__ import annotations

import ast
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Rule Modifiers — parameterize gold plan generation for adaptation phases
# ---------------------------------------------------------------------------

@dataclass
class RuleModifiers:
    """Encodes active rule changes for gold plan generation.

    scarcity: If True, Chain-type (low_resource) management does NOT persist
              within a mission. Ring-type (bad_quality) management still persists.
    integrate_after_input: If True, emit "INTEGRATE" after every *_input resolution
                          (both direct token and subgoal expansion).
    log_after_management: If True, emit "LOG" after every zone management completion
                         (including the top-level target zone).
    """
    scarcity: bool = False
    integrate_after_input: bool = False
    log_after_management: bool = False


@dataclass
class ParseResult:
    actions: list[str]
    parse_ok: bool
    format_strict_ok: bool
    parse_mode: str
    parser_error: str


@dataclass
class MissionResult:
    mission_success: bool   # Primary metric: predicted == gold exactly (optimal match)
    plan_complete: bool     # Secondary: predicted contains full gold as a prefix (allows extra trailing tokens)
    step_validity: float    # Partial-credit: fraction of gold steps correctly predicted in order
    failure_label: str
    first_error_position: int
    optimal_plan_length: int


def compile_environment_runtime(environment_spec: dict[str, Any]) -> dict[str, Any]:
    zones = list(environment_spec["zones"])
    input_action_by_zone = dict(environment_spec["input_action_by_zone"])
    dependencies = dict(environment_spec["dependencies"])
    primitive_template_by_zone: dict[str, list[Any]] = {}
    for zone in zones:
        converted_steps: list[Any] = []
        for step in environment_spec["primitive_template_by_zone"][zone]:
            step_text = str(step)
            if step_text.endswith("_INPUT"):
                dep_zone = dependencies[zone]
                converted_steps.append(("INPUT", dep_zone))
            else:
                converted_steps.append(step_text.upper())
        primitive_template_by_zone[zone] = converted_steps
    return {
        "zones": zones,
        "input_action_by_zone": input_action_by_zone,
        "primitive_template_by_zone": primitive_template_by_zone,
    }


def build_canonical_action_regex(allowed_tokens: list[str]) -> re.Pattern[str]:
    escaped = "|".join(re.escape(token) for token in sorted(allowed_tokens, key=len, reverse=True))
    return re.compile(rf"\b({escaped})\b", flags=re.IGNORECASE)


def parse_condition(condition: str) -> tuple[str, int]:
    label = str(condition).strip()
    if label == "ideal":
        return ("ideal", 0)
    if label == "low_resource":
        return ("low_resource", 1)
    if label.startswith("bad_quality_"):
        return ("bad_quality", int(label.split("_")[-1]))
    if label == "bad_quality":
        raise ValueError("Symbolic 'bad_quality' must be concretized before simulation.")
    raise ValueError(f"Unsupported condition: {condition}")


def build_initial_state(row: dict[str, Any], zones: list[str]) -> dict[str, tuple[str, int]]:
    return {zone: parse_condition(row[f"{zone}_condition"]) for zone in zones}


def _expand_zone_goal(
    zone: str,
    state: dict[str, tuple[str, int]],
    *,
    primitive_template_by_zone: dict[str, list[Any]],
    input_action_by_zone: dict[str, str],
    rules: RuleModifiers | None = None,
) -> list[str]:
    _rules = rules or RuleModifiers()
    condition_kind, level = state[zone]
    if condition_kind == "ideal":
        return []
    primitive = primitive_template_by_zone[zone]
    budget = len(primitive) if condition_kind == "low_resource" else level
    output: list[str] = []
    for step_index in range(budget):
        step = primitive[step_index % len(primitive)]
        if isinstance(step, tuple):
            _, dependency_zone = step
            if state[dependency_zone][0] == "ideal":
                output.append(input_action_by_zone[dependency_zone])
            else:
                output.extend(
                    _expand_zone_goal(
                        dependency_zone,
                        state,
                        primitive_template_by_zone=primitive_template_by_zone,
                        input_action_by_zone=input_action_by_zone,
                        rules=_rules,
                    )
                )
            # Rule 1: INTEGRATE after every *_input resolution
            if _rules.integrate_after_input:
                output.append("INTEGRATE")
        else:
            output.append(step)

    # State persistence: update zone to ideal
    if _rules.scarcity and condition_kind == "low_resource":
        # Rule 3: Chain (low_resource) management does NOT persist
        pass  # state[zone] remains unchanged
    else:
        state[zone] = ("ideal", 0)

    # Rule 2: LOG after zone management completion
    if _rules.log_after_management:
        output.append("LOG")

    return output


def build_gold_plan_for_row(
    row: dict[str, Any],
    env_runtime: dict[str, Any],
    rules: RuleModifiers | None = None,
) -> list[str]:
    state = build_initial_state(row, env_runtime["zones"])
    return _expand_zone_goal(
        str(row["target_zone"]),
        state,
        primitive_template_by_zone=env_runtime["primitive_template_by_zone"],
        input_action_by_zone=env_runtime["input_action_by_zone"],
        rules=rules,
    )


def _extract_json_object(raw_text: str) -> str | None:
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw_text, flags=re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    start = raw_text.find("{")
    if start == -1:
        return None
    depth = 0
    for index, char in enumerate(raw_text[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw_text[start : index + 1].strip()
    return None


def _normalize_actions(actions: Any, allowed_tokens: set[str]) -> list[str]:
    if not isinstance(actions, list) or not actions:
        raise ValueError("'actions' must be non-empty list")
    normalized = [str(token).strip().upper() for token in actions]
    unknown = [token for token in normalized if token not in allowed_tokens]
    if unknown:
        raise ValueError(f"Unknown action tokens: {sorted(set(unknown))}")
    return normalized


def parse_actions_from_raw_text(raw_text: Any, allowed_tokens: list[str], canonical_action_regex: re.Pattern[str]) -> ParseResult:
    allowed_set = set(allowed_tokens)
    payload_text = "" if raw_text is None else str(raw_text)
    strict_error = ""
    extracted_error = ""
    try:
        payload = json.loads(payload_text)
        if not isinstance(payload, dict) or "actions" not in payload:
            raise ValueError("Missing top-level key 'actions'")
        return ParseResult(_normalize_actions(payload["actions"], allowed_set), True, True, "json_strict", "")
    except Exception as error:
        strict_error = str(error)

    extracted = _extract_json_object(payload_text)
    if extracted is not None:
        try:
            payload = json.loads(extracted)
            if not isinstance(payload, dict) or "actions" not in payload:
                raise ValueError("Missing top-level key 'actions'")
            return ParseResult(_normalize_actions(payload["actions"], allowed_set), True, False, "json_extracted", "")
        except Exception as error:
            extracted_error = str(error)

    # Array literal fallback (new_implementations_todo.md §6.2): models sometimes
    # wrap the answer in prose like "the sequence is [\"AIR\", \"PLANT\"]".
    # Pull the first [...] block and literal-eval it. Tolerates single quotes
    # and bare tokens that JSON would reject.
    array_match = re.search(r"\[[^\[\]]*\]", payload_text)
    literal_error = ""
    if array_match is not None:
        array_text = array_match.group(0)
        try:
            parsed = ast.literal_eval(array_text)
            return ParseResult(
                _normalize_actions(parsed, allowed_set), True, False, "array_literal", ""
            )
        except Exception as error:
            literal_error = str(error)

    salvaged = [match.group(1).upper() for match in canonical_action_regex.finditer(payload_text)]
    if salvaged:
        return ParseResult(salvaged, True, False, "token_salvage", "")

    parser_error = f"strict: {strict_error}"
    if extracted_error:
        parser_error += f" | extracted: {extracted_error}"
    if literal_error:
        parser_error += f" | literal: {literal_error}"
    return ParseResult([], False, False, "unparseable", parser_error)


def evaluate_actions_against_gold(predicted_actions: list[str], gold_actions: list[str], target_zone_condition: str) -> MissionResult:
    min_length = min(len(predicted_actions), len(gold_actions))
    valid_prefix = 0
    for index in range(min_length):
        if predicted_actions[index] != gold_actions[index]:
            break
        valid_prefix += 1
    # plan_complete: all gold tokens appear as a prefix (extra trailing tokens allowed)
    plan_complete = valid_prefix == len(gold_actions)
    # mission_success (primary): exactly matches the optimal gold sequence — no missing, no extra
    mission_success = plan_complete and len(predicted_actions) == len(gold_actions)

    if mission_success:
        return MissionResult(True, True, 1.0, "NONE", -1, len(gold_actions))

    if plan_complete:
        # Gold prefix satisfied but extra trailing tokens were emitted — model didn't learn when to stop
        return MissionResult(False, True, 1.0, "REDUNDANT_ACTIONS", len(gold_actions), len(gold_actions))

    if valid_prefix == len(predicted_actions):
        # Predicted is a strict prefix of gold — stopped too early
        failure_label = "WRONG_RING_REPEAT_COUNT" if str(target_zone_condition).startswith("bad_quality_") else "TARGET_NOT_IDEAL"
        return MissionResult(False, False, valid_prefix / max(1, len(gold_actions)), failure_label, valid_prefix, len(gold_actions))

    # Predicted deviated at some interior position
    return MissionResult(False, False, valid_prefix / max(1, len(gold_actions)), "ILLEGAL_ACTION_ORDER", valid_prefix, len(gold_actions))


def concretize_bad_quality(row: dict[str, Any], zones: list[str], *, low: int, high: int, rng: random.Random) -> tuple[dict[str, Any], dict[str, int]]:
    concrete = dict(row)
    sampled_levels = {f"sampled_{zone}_level": 0 for zone in zones}
    for zone in zones:
        condition_key = f"{zone}_condition"
        if str(concrete[condition_key]) == "bad_quality":
            sampled_level = int(rng.randint(low, high))
            concrete[condition_key] = f"bad_quality_{sampled_level}"
            sampled_levels[f"sampled_{zone}_level"] = sampled_level
    return concrete, sampled_levels
