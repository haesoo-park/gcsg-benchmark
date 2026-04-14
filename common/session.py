from __future__ import annotations

import json
import random
from typing import Any, Callable

import pandas as pd
from pydantic import BaseModel, Field

from dataclasses import dataclass, field

from .core import (
    MissionResult,
    ParseResult,
    RuleModifiers,
    build_canonical_action_regex,
    build_gold_plan_for_row,
    concretize_bad_quality,
    evaluate_actions_against_gold,
    parse_actions_from_raw_text,
)
from .metrics import summarize_learning_metrics
from .prompts import (
    CONTEXT_INIT_LABELS,
    CONTEXT_INIT_OPTIONS,
    PROMPT_STYLE_CONFIG_OPTIONS,
    build_instruction_block,
    build_mission_prompt,
    build_mission_text,
    resolve_prompt_style,
)


class ActionPlan(BaseModel):
    actions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline robustness — mechanical API failures vs cognitive errors
# ---------------------------------------------------------------------------
# new_implementations_todo.md §6.1: a model hitting a Max-Token or
# Context-Exceeded hard limit during an 80+ mission adaptation loop should
# register as API_FAIL (nulling the task's score) rather than as a 0.0
# mission accuracy. We detect this by scanning the exception text for
# provider-agnostic substrings.

class APIFailError(RuntimeError):
    """Raised when an LLM call fails for mechanical (not cognitive) reasons.

    Propagates out of session loops so notebooks can return a null score
    instead of silently averaging the failure into the cognitive metric.
    """


_API_FAIL_MARKERS = (
    "context length",
    "context_length",
    "context window",
    "maximum context",
    "max context",
    "prompt is too long",
    "too many tokens",
    "max_tokens",
    "token limit",
    "rate limit",
    "rate_limit",
)


def _is_api_fail(error: BaseException) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    return any(marker in text for marker in _API_FAIL_MARKERS)


# ---------------------------------------------------------------------------
# Schedule builders (unchanged from v1)
# ---------------------------------------------------------------------------

def build_interleaved_schedule(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    train_block_size: int,
) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    train_index = 0
    test_index = 0
    while train_index < len(train_rows):
        for _ in range(train_block_size):
            if train_index >= len(train_rows):
                break
            schedule.append({"row": train_rows[train_index], "inject_feedback": True, "stage": "train"})
            train_index += 1
        if test_index < len(test_rows):
            schedule.append({"row": test_rows[test_index], "inject_feedback": False, "stage": "probe"})
            test_index += 1
    while test_index < len(test_rows):
        schedule.append({"row": test_rows[test_index], "inject_feedback": False, "stage": "probe_tail"})
        test_index += 1
    return schedule


def build_end_block_test_schedule(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    for row in train_rows:
        schedule.append({"row": row, "inject_feedback": True, "stage": "train"})
    for row in test_rows:
        schedule.append({"row": row, "inject_feedback": False, "stage": "test_end"})
    return schedule


# ---------------------------------------------------------------------------
# Main session
# ---------------------------------------------------------------------------

def run_learning_session(
    *,
    llm: Any,
    kbench: Any,
    task_df: pd.DataFrame,
    task_config: dict[str, Any],
    env_runtime: dict[str, Any],
    context_init_option: str,
    prompt_style_config: str,
    probe_schedule_mode: str = "end_block_test",
    include_output_format_prompt: bool = True,
    feedback_mode: str = "feedback_and_ack",
    bad_quality_level_range_by_split: dict[str, dict[str, int]] | None = None,
    early_stop_perfect_train_window: int = 0,
    early_stop_perfect_probe_window: int = 0,
    collect_rule_quiz: bool = False,
    rule_quiz_prompt: str | None = None,
    enable_live_mission_log: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # ── validate inputs ──────────────────────────────────────────────────
    ctx_option = str(context_init_option).strip().lower()
    if ctx_option not in CONTEXT_INIT_OPTIONS:
        valid = ", ".join(sorted(CONTEXT_INIT_OPTIONS.keys()))
        raise ValueError(f"context_init_option must be one of {valid}")

    style_cfg = str(prompt_style_config).strip().lower()
    if style_cfg not in PROMPT_STYLE_CONFIG_OPTIONS:
        valid = ", ".join(sorted(PROMPT_STYLE_CONFIG_OPTIONS.keys()))
        raise ValueError(f"prompt_style_config must be one of {valid}")

    schedule_mode = str(probe_schedule_mode).strip().lower()
    if schedule_mode not in {"interleaved", "end_block_test"}:
        raise ValueError("probe_schedule_mode must be 'interleaved' or 'end_block_test'")

    fb_mode = str(feedback_mode).strip().lower()
    if fb_mode not in {"feedback_only", "feedback_and_ack", "none"}:
        raise ValueError("feedback_mode must be 'feedback_only', 'feedback_and_ack', or 'none'")

    # ── split configuration ──────────────────────────────────────────────
    train_split_name = task_config["protocol"]["train_split_name"]
    test_split_name = task_config["protocol"]["test_split_name"]
    default_bq_low = int(task_config["bad_quality_level_range"]["min"])
    default_bq_high = int(task_config["bad_quality_level_range"]["max"])

    split_bad_quality_ranges: dict[str, tuple[int, int]] = {
        train_split_name: (default_bq_low, default_bq_high),
        test_split_name: (default_bq_low, default_bq_high),
    }
    if bad_quality_level_range_by_split is not None:
        for split_name in (train_split_name, test_split_name):
            split_config = bad_quality_level_range_by_split.get(split_name)
            if split_config is None:
                continue
            low = int(split_config["min"])
            high = int(split_config["max"])
            if low > high:
                raise ValueError(f"Invalid bad_quality range for split '{split_name}': min {low} > max {high}")
            split_bad_quality_ranges[split_name] = (low, high)

    # ── prepare rows and schedule ────────────────────────────────────────
    train_df = task_df[task_df["mission_split"] == train_split_name].copy()
    test_df = task_df[task_df["mission_split"] == test_split_name].copy()
    train_rows = train_df.sort_values(["goal_level", "split_index"]).to_dict(orient="records")
    test_rows = test_df.sort_values(["goal_level", "split_index"]).to_dict(orient="records")

    rng_schedule = random.Random(int(task_config["seeds"]["schedule_seed"]))
    rng_runtime = random.Random(int(task_config["seeds"]["runtime_sampling_seed"]))
    rng_schedule.shuffle(train_rows)
    rng_schedule.shuffle(test_rows)

    if schedule_mode == "interleaved":
        schedule = build_interleaved_schedule(
            train_rows,
            test_rows,
            int(task_config["protocol"]["probe_after_train_missions"]),
        )
    else:
        schedule = build_end_block_test_schedule(train_rows, test_rows)

    # ── allowed tokens & regex ───────────────────────────────────────────
    zones = list(env_runtime["zones"])
    allowed_tokens = sorted(set(env_runtime["input_action_by_zone"].values()) | {
        step
        for steps in env_runtime["primitive_template_by_zone"].values()
        for step in steps
        if isinstance(step, str)
    })
    canonical_action_regex = build_canonical_action_regex(allowed_tokens)

    # ── build static instruction block ───────────────────────────────────
    instruction_block = build_instruction_block(
        context_init_option=ctx_option,
        include_output_format_prompt=include_output_format_prompt,
    )

    # ── mission loop ─────────────────────────────────────────────────────
    traces: list[dict[str, Any]] = []
    recent_train_success: list[bool] = []
    recent_probe_success: list[bool] = []
    running_total_count = 0
    running_total_success = 0
    running_train_count = 0
    running_train_success = 0
    running_probe_count = 0
    running_probe_success = 0
    quiz_response = ""
    quiz_prompt_used = ""

    with kbench.chats.new(f"learning_session_{task_config['task_id']}_{ctx_option}"):
        # ── send instruction block once as session prologue ──────────
        try:
            llm.prompt(instruction_block)
        except Exception as prologue_err:
            if _is_api_fail(prologue_err):
                raise APIFailError(
                    f"mechanical API failure (prologue): {prologue_err}"
                ) from prologue_err

        for run_position, item in enumerate(schedule, start=1):
            is_train = bool(item["inject_feedback"])

            # ── early stop ───────────────────────────────────────────
            if (
                early_stop_perfect_train_window > 0
                and is_train
                and len(recent_train_success) >= early_stop_perfect_train_window
                and all(recent_train_success[-early_stop_perfect_train_window:])
            ):
                continue
            if (
                early_stop_perfect_probe_window > 0
                and not is_train
                and len(recent_probe_success) >= early_stop_perfect_probe_window
                and all(recent_probe_success[-early_stop_perfect_probe_window:])
            ):
                continue

            # ── concretize bad_quality ────────────────────────────────
            base_row = item["row"]
            active_split_name = train_split_name if is_train else test_split_name
            sampling_low, sampling_high = split_bad_quality_ranges[active_split_name]
            concrete_row, sampled_levels = concretize_bad_quality(
                base_row, zones, low=sampling_low, high=sampling_high, rng=rng_runtime,
            )

            # ── resolve prompt style ─────────────────────────────────
            prompt_style = resolve_prompt_style(style_cfg, is_train, rng_runtime)

            # ── build prompt (mission text only — prologue already sent) ──
            prompt_text = build_mission_text(
                row=concrete_row,
                prompt_style=prompt_style,
            )

            # ── call LLM ─────────────────────────────────────────────
            raw_response = ""
            parse_result = ParseResult([], False, False, "uninitialized", "")
            try:
                structured = llm.prompt(prompt_text, schema=ActionPlan)
                actions = [str(token).strip().upper() for token in structured.actions]
                if not actions:
                    raise ValueError("Schema response had empty actions.")
                raw_response = json.dumps({"actions": actions})
                parse_result = ParseResult(actions, True, True, "schema", "")
            except Exception as schema_error:
                if _is_api_fail(schema_error):
                    raise APIFailError(
                        f"mechanical API failure (schema call): {schema_error}"
                    ) from schema_error
                try:
                    raw_response = llm.prompt(prompt_text)
                    parse_result = parse_actions_from_raw_text(raw_response, allowed_tokens, canonical_action_regex)
                    if not parse_result.parse_ok:
                        parse_result.parser_error = f"schema: {schema_error} | {parse_result.parser_error}"
                except Exception as fallback_error:
                    if _is_api_fail(fallback_error):
                        raise APIFailError(
                            f"mechanical API failure (raw fallback): {fallback_error}"
                        ) from fallback_error
                    parse_result = ParseResult([], False, False, "llm_call_failed", f"{schema_error} | {fallback_error}")

            # ── score ─────────────────────────────────────────────────
            gold_actions = build_gold_plan_for_row(concrete_row, env_runtime)
            mission_result = evaluate_actions_against_gold(
                parse_result.actions,
                gold_actions,
                concrete_row[f"{concrete_row['target_zone']}_condition"],
            )
            if not parse_result.parse_ok:
                mission_result = MissionResult(False, False, 0.0, "FORMAT_ERROR", 0, max(1, len(gold_actions)))

            # ── training feedback ─────────────────────────────────────
            feedback_ack = ""
            feedback_payload = ""
            if is_train and fb_mode != "none":
                _fb_prefix = (
                    "Correct." if mission_result.mission_success else "Incorrect."
                )
                _fb_body = (
                    f" The optimal action sequence for that "
                    f"mission was: {json.dumps(gold_actions)}"
                )
                if fb_mode == "feedback_and_ack":
                    feedback_payload = _fb_prefix + _fb_body + "\nAcknowledge with: OK"
                else:
                    feedback_payload = _fb_prefix + _fb_body
                try:
                    feedback_ack = llm.prompt(feedback_payload)
                except Exception as feedback_error:
                    feedback_ack = f"FEEDBACK_ACK_FAILED: {feedback_error}"

            if is_train:
                recent_train_success.append(bool(mission_result.mission_success))
            else:
                recent_probe_success.append(bool(mission_result.mission_success))

            # ── trace row ─────────────────────────────────────────────
            trace: dict[str, Any] = {
                "task_id": base_row["task_id"],
                "mission_id": base_row["mission_id"],
                "source_mission_id": base_row["source_mission_id"],
                "template_id": base_row.get("template_id", ""),
                "goal_level": int(base_row["goal_level"]),
                "target_zone": base_row["target_zone"],
                "test_category": str(base_row.get("test_category", "") or ""),
                "mission_split": base_row["mission_split"],
                "split_index": int(base_row["split_index"]),
                "prompt_style": prompt_style,
                "context_init_option": ctx_option,
                "context_init_label": CONTEXT_INIT_LABELS[ctx_option],
                "prompt_style_config": style_cfg,
                "stage": item["stage"],
                "feedback_injected": is_train,
                "feedback_mode": fb_mode,
                "feedback_ack": feedback_ack,
                "mission_success": mission_result.mission_success,
                "plan_complete": mission_result.plan_complete,
                "step_validity": mission_result.step_validity,
                "format_success": parse_result.format_strict_ok,
                "failure_label": mission_result.failure_label,
                "first_error_position": mission_result.first_error_position,
                "optimal_plan_length": mission_result.optimal_plan_length,
                "parse_ok": parse_result.parse_ok,
                "parse_mode": parse_result.parse_mode,
                "parser_error": parse_result.parser_error,
                "predicted_actions": parse_result.actions,
                "optimal_actions": gold_actions,
                "prompt_text": prompt_text,
                "feedback_payload": feedback_payload,
                "raw_response": raw_response,
                "run_position": run_position,
            }
            for zone in zones:
                trace[f"{zone}_condition"] = concrete_row[f"{zone}_condition"]
                trace[f"sampled_{zone}_level"] = sampled_levels[f"sampled_{zone}_level"]
            traces.append(trace)

            # ── running counters ──────────────────────────────────────
            running_total_count += 1
            running_total_success += int(bool(mission_result.mission_success))
            if is_train:
                running_train_count += 1
                running_train_success += int(bool(mission_result.mission_success))
            else:
                running_probe_count += 1
                running_probe_success += int(bool(mission_result.mission_success))

            # ── live mission log ──────────────────────────────────────
            if enable_live_mission_log:
                success_flag = "T" if mission_result.mission_success else "F"
                train_acc = running_train_success / running_train_count if running_train_count else 0.0
                probe_acc = running_probe_success / running_probe_count if running_probe_count else 0.0
                ps_short = prompt_style[0]
                _stage = item["stage"]
                print(
                    f"[{ctx_option}] {base_row['mission_id']} | "
                    f"{base_row.get('template_id', '')} | "
                    f"prompt-style={ps_short} | "
                    f"success={success_flag} | "
                    f"train_acc={train_acc:.3f} probe_acc={probe_acc:.3f}"
                )

            # ── progress callback ─────────────────────────────────────
            if progress_callback is not None:
                progress_payload = {
                    "trace": trace,
                    "running_total_accuracy": (running_total_success / running_total_count) if running_total_count else 0.0,
                    "running_train_accuracy": (running_train_success / running_train_count) if running_train_count else 0.0,
                    "running_probe_accuracy": (running_probe_success / running_probe_count) if running_probe_count else 0.0,
                    "running_total_count": running_total_count,
                    "running_train_count": running_train_count,
                    "running_probe_count": running_probe_count,
                }
                try:
                    progress_callback(progress_payload)
                except Exception:
                    pass

        # ── rule quiz ─────────────────────────────────────────────────
        if collect_rule_quiz:
            quiz_prompt_used = (
                str(rule_quiz_prompt).strip()
                if rule_quiz_prompt is not None and str(rule_quiz_prompt).strip()
                else (
                    "You just completed a sequence of missions in this environment.\n"
                    "Describe the underlying environment rules you inferred from training and testing.\n"
                    "Be explicit about zone dependencies, action ordering, and condition-to-action mapping.\n"
                    "Also summarize where you are uncertain."
                )
            )
            try:
                quiz_response = str(llm.prompt(quiz_prompt_used))
            except Exception as quiz_error:
                quiz_response = f"QUIZ_PROMPT_FAILED: {quiz_error}"

    # ── aggregate ─────────────────────────────────────────────────────
    trace_df = pd.DataFrame(traces)
    metrics = summarize_learning_metrics(
        trace_df,
        task_id=str(task_config["task_id"]),
        prompt_config_key=ctx_option,
        prompt_config_label=CONTEXT_INIT_LABELS[ctx_option],
        prompt_config_depths=[],  # not applicable in v2
        context_init_style=ctx_option,
    )
    if collect_rule_quiz:
        metrics["rule_quiz_prompt"] = quiz_prompt_used
        metrics["rule_quiz_response"] = quiz_response
    return trace_df, metrics


# ═══════════════════════════════════════════════════════════════════════════
# Phased Session — multi-phase protocol with rule adaptation & retention
# ═══════════════════════════════════════════════════════════════════════════

# ── Rule injection / reversal prompts ─────────────────────────────────────

RULE_PROMPTS: dict[str, dict[str, str]] = {
    "scarcity": {
        # Full injection — used for the ceiling condition (explicit rule description)
        "injection": (
            "IMPORTANT UPDATE — starting from the next mission, we are "
            "operating under scarcity conditions.\n\n"
            "Previously, when you managed a dependency zone, its condition "
            "changed to ideal for the remainder of the mission. This is no "
            "longer always the case.\n\n"
            "New rule: Managing a dependency zone that is low on resource "
            "NO LONGER changes its condition to ideal for the remainder of "
            "that mission. Every time you need input from a dependency zone "
            "that is low on resource, you must manage it each time you need "
            "input from it.\n\n"
            "However, managing a dependency zone that has bad quality "
            "resource STILL changes its condition to ideal for the remainder "
            "of the mission, as before.\n\n"
            "This rule applies to all following missions until I say otherwise.\n"
            "Acknowledge with: OK"
        ),
        # Baseline injection — used for learning/control conditions (change signal only, no rule description)
        "injection_baseline": (
            "IMPORTANT UPDATE — starting from the next mission, we are "
            "operating under scarcity conditions, and the rules for this "
            "phase have changed.\n"
            "Acknowledge with: OK"
        ),
        "reversal": (
            "IMPORTANT UPDATE — the scarcity conditions have been lifted. "
            "Starting from the next mission, the ecosystem returns to normal "
            "operation.\n\n"
            "When you manage any dependency zone — whether it was low on "
            "resource or had bad quality — its condition returns to ideal for "
            "the remainder of the mission, just as it was before the scarcity "
            "period.\n\n"
            "This applies to all following missions.\n"
            "Acknowledge with: OK"
        ),
        # Baseline reversal — used for learning/control conditions
        "reversal_baseline": (
            "IMPORTANT UPDATE — the scarcity conditions have been lifted. "
            "Starting from the next mission, the ecosystem returns to normal "
            "operation.\n"
            "Acknowledge with: OK"
        ),
    },
    "integrate": {
        # Full injection — used for the ceiling condition (explicit rule description)
        "injection": (
            "IMPORTANT UPDATE — starting from the next mission, communication "
            "between zones has weakened.\n\n"
            "New rule: Right after obtaining input from a dependency zone "
            "— whether it is a single resource action (for an ideal zone) or "
            "a full management sequence (for a non-ideal zone) — you must "
            "perform the \"INTEGRATE\" action.\n\n"
            "You now have a new action available: \"INTEGRATE\"\n\n"
            "This applies to all following missions until I say otherwise.\n"
            "Acknowledge with: OK"
        ),
        # Baseline injection — used for learning/control conditions (change signal only)
        "injection_baseline": (
            "IMPORTANT UPDATE — starting from the next mission, communication "
            "between zones has weakened, and the rules for this phase have "
            "changed.\n"
            "Acknowledge with: OK"
        ),
        "reversal": (
            "IMPORTANT UPDATE — communication between zones has been fully "
            "restored. Starting from the next mission, you no longer need to "
            "perform the \"INTEGRATE\" action after dependency inputs.\n\n"
            "Remove \"INTEGRATE\" from your action sequences and return to "
            "the original action set.\n\n"
            "This applies to all following missions.\n"
            "Acknowledge with: OK"
        ),
        # Baseline reversal — used for learning/control conditions
        "reversal_baseline": (
            "IMPORTANT UPDATE — communication between zones has been fully "
            "restored. Starting from the next mission, the ecosystem returns "
            "to normal operation.\n"
            "Acknowledge with: OK"
        ),
    },
    "log": {
        "injection": (
            "IMPORTANT UPDATE — starting from the next mission, management "
            "logging is required.\n\n"
            "New rule: Every time you complete managing a zone (changing its "
            "status to ideal), you must perform the \"LOG\" action immediately "
            "after the last action of that zone's management.\n\n"
            "You now have a new action available: \"LOG\"\n\n"
            "This applies to all following missions until I say otherwise.\n"
            "Acknowledge with: OK"
        ),
        "reversal": (
            "IMPORTANT UPDATE — the logging requirement has been removed. "
            "Starting from the next mission, you no longer need to perform "
            "the \"LOG\" action after zone management.\n\n"
            "Remove \"LOG\" from your action sequences and return to the "
            "original action set.\n\n"
            "This applies to all following missions.\n"
            "Acknowledge with: OK"
        ),
    },
}

RULE_TYPE_TO_MODIFIERS: dict[str, RuleModifiers] = {
    "scarcity": RuleModifiers(scarcity=True),
    "integrate": RuleModifiers(integrate_after_input=True),
    "log": RuleModifiers(log_after_management=True),
}


@dataclass
class PhaseSpec:
    """Specification for one phase of a multi-phase experiment.

    Attributes:
        name: Unique phase identifier (e.g. 'original_train', 'adapted_probe').
        stage_label: Value used in trace['stage'] column.
        mission_rows: Mission row dicts to present in this phase.
        inject_feedback: Whether to provide corrective feedback after each mission.
        rules: Active RuleModifiers for gold plan generation (None = standard rules).
        feedback_mode: How feedback is delivered ('feedback_and_ack', 'feedback_only', 'none').
        pre_phase_prompt: Optional message to send to the LLM before this phase starts.
    """
    name: str
    stage_label: str
    mission_rows: list[dict[str, Any]]
    inject_feedback: bool
    rules: RuleModifiers | None = None
    feedback_mode: str = "feedback_and_ack"
    pre_phase_prompt: str | None = None
    bad_quality_range: dict[str, int] | None = None  # {"min": N, "max": M} — overrides session-level range
    early_stop_perfect_window: int = 0  # if > 0, skip remaining missions after N consecutive correct
    # ── T8 transfer support ──────────────────────────────────────────────
    env_runtime_override: dict[str, Any] | None = None  # use different env_runtime for this phase
    mission_text_builder: Callable[[dict[str, Any], str], str] | None = None  # custom (row, style) → text


def run_phased_session(
    *,
    llm: Any,
    kbench: Any,
    phases: list[PhaseSpec],
    task_config: dict[str, Any],
    env_runtime: dict[str, Any],
    context_init_option: str = "c",
    prompt_style_config: str = "a",
    include_output_format_prompt: bool = True,
    bad_quality_level_range: dict[str, int] | None = None,
    # Retention probes
    retention_after_phases: list[str] | None = None,
    retention_count: int = 5,
    # Structured quiz
    structured_quiz_questions: list[dict[str, Any]] | None = None,
    quiz_after_phases: list[str] | None = None,
    # Misc
    enable_live_mission_log: bool = False,
    session_label: str = "",
) -> dict[str, Any]:
    """Run a multi-phase experiment within a single chat context.

    All phases share one continuous conversation, so the LLM retains context
    across rule changes, training, probing, and retention tests.

    Args:
        phases: Ordered list of PhaseSpec objects defining the experiment.
        retention_after_phases: Phase names after which to insert retention probes
            (re-presents concretized missions from the first train phase).
        retention_count: Number of early training missions to re-probe.
        structured_quiz_questions: If provided, runs structured quiz at quiz points.
        quiz_after_phases: Phase names after which to run the structured quiz.

    Returns:
        Dict with keys:
          - 'phase_traces': {phase_name: pd.DataFrame}
          - 'all_traces': pd.DataFrame (concatenated across all phases)
          - 'phase_metrics': {phase_name: dict}
          - 'retention_traces': {insertion_point: pd.DataFrame}
          - 'quiz_results': {insertion_point: list[dict]}
          - 'saved_train_missions': list[dict] (concretized missions from first train phase)
    """
    from .quiz import run_structured_quiz, summarize_quiz_results

    # ── validate inputs ──────────────────────────────────────────────────
    ctx_option = str(context_init_option).strip().lower()
    if ctx_option not in CONTEXT_INIT_OPTIONS:
        raise ValueError(f"context_init_option must be one of {sorted(CONTEXT_INIT_OPTIONS.keys())}")
    style_cfg = str(prompt_style_config).strip().lower()
    if style_cfg not in PROMPT_STYLE_CONFIG_OPTIONS:
        raise ValueError(f"prompt_style_config must be one of {sorted(PROMPT_STYLE_CONFIG_OPTIONS.keys())}")

    retention_after = set(retention_after_phases or [])
    quiz_after = set(quiz_after_phases or [])

    # ── common resources ─────────────────────────────────────────────────
    zones = list(env_runtime["zones"])
    base_allowed_tokens = sorted(set(env_runtime["input_action_by_zone"].values()) | {
        step
        for steps in env_runtime["primitive_template_by_zone"].values()
        for step in steps
        if isinstance(step, str)
    })
    canonical_action_regex = build_canonical_action_regex(base_allowed_tokens)

    bq_low = int((bad_quality_level_range or task_config["bad_quality_level_range"])["min"])
    bq_high = int((bad_quality_level_range or task_config["bad_quality_level_range"])["max"])

    instruction_block = build_instruction_block(
        context_init_option=ctx_option,
        include_output_format_prompt=include_output_format_prompt,
    )

    rng_runtime = random.Random(int(task_config["seeds"]["runtime_sampling_seed"]))

    # ── results containers ───────────────────────────────────────────────
    phase_traces: dict[str, pd.DataFrame] = {}
    phase_metrics: dict[str, dict[str, Any]] = {}
    retention_traces: dict[str, pd.DataFrame] = {}
    quiz_results: dict[str, Any] = {}
    saved_train_missions: list[dict[str, Any]] = []
    all_trace_rows: list[dict[str, Any]] = []
    global_position = 0

    task_id = str(task_config["task_id"])
    chat_label = session_label or f"phased_{task_id}_{ctx_option}"

    # ── open single chat context ─────────────────────────────────────────
    with kbench.chats.new(chat_label):
        # ── send instruction block once as session prologue ──────────
        try:
            llm.prompt(instruction_block)
        except Exception as prologue_err:
            if _is_api_fail(prologue_err):
                raise APIFailError(
                    f"mechanical API failure (prologue): {prologue_err}"
                ) from prologue_err

        for phase in phases:
            # ── phase boundary ──────────────────────────────────────
            if enable_live_mission_log:
                _cond_lbl = session_label or ctx_option
                _n_missions = len(phase.mission_rows) if phase.mission_rows else 0
                print(f"{'─' * 60}")
                print(f"{_cond_lbl} [{phase.name}]  ({_n_missions} missions)")
                print(f"{'─' * 60}")

            # ── pre-phase prompt (rule injection / reversal) ─────────
            if phase.pre_phase_prompt:
                try:
                    ack = llm.prompt(phase.pre_phase_prompt)
                    if enable_live_mission_log:
                        _cond_lbl = session_label or ctx_option
                        print(f"{_cond_lbl} [{phase.name}] pre-phase prompt sent")
                except Exception as e:
                    if enable_live_mission_log:
                        _cond_lbl = session_label or ctx_option
                        print(f"{_cond_lbl} [{phase.name}] pre-phase prompt FAILED: {e}")

            if not phase.mission_rows:
                # Phase with only a pre-phase prompt (e.g. pure rule injection)
                continue

            # ── determine env resources for this phase ───────────────
            if phase.env_runtime_override is not None:
                phase_env = phase.env_runtime_override
                phase_zones = list(phase_env["zones"])
                phase_allowed_tokens = sorted(
                    set(phase_env["input_action_by_zone"].values()) | {
                        step
                        for steps in phase_env["primitive_template_by_zone"].values()
                        for step in steps
                        if isinstance(step, str)
                    }
                )
            else:
                phase_env = env_runtime
                phase_zones = zones
                phase_allowed_tokens = list(base_allowed_tokens)

            if phase.rules and phase.rules.integrate_after_input:
                phase_allowed_tokens.append("INTEGRATE")
            if phase.rules and phase.rules.log_after_management:
                phase_allowed_tokens.append("LOG")

            allowed_tokens = phase_allowed_tokens
            phase_action_regex = build_canonical_action_regex(allowed_tokens)

            fb_mode = str(phase.feedback_mode).strip().lower()
            phase_trace_rows: list[dict[str, Any]] = []
            _recent_success: list[bool] = []
            _pending_feedback: str = ""  # v0.8.0: combine feedback with next mission

            for mission_idx, row in enumerate(phase.mission_rows):
                # ── early stop check ────────────────────────────────
                if (
                    phase.early_stop_perfect_window > 0
                    and len(_recent_success) >= phase.early_stop_perfect_window
                    and all(_recent_success[-phase.early_stop_perfect_window:])
                ):
                    if enable_live_mission_log:
                        _cond_lbl = session_label or ctx_option
                        print(
                            f"{_cond_lbl} [{phase.name}] EARLY STOP — last "
                            f"{phase.early_stop_perfect_window} correct, "
                            f"skipping {len(phase.mission_rows) - mission_idx}"
                        )
                    break
                global_position += 1

                # ── concretize bad_quality ──────────────────────��─────
                phase_bq_low = int(phase.bad_quality_range["min"]) if phase.bad_quality_range else bq_low
                phase_bq_high = int(phase.bad_quality_range["max"]) if phase.bad_quality_range else bq_high
                concrete_row, sampled_levels = concretize_bad_quality(
                    row, phase_zones, low=phase_bq_low, high=phase_bq_high, rng=rng_runtime,
                )

                # Save concretized missions from first train phase for retention
                if phase.inject_feedback and len(saved_train_missions) < retention_count:
                    saved_train_missions.append(dict(concrete_row))

                # ── resolve prompt style ─────────────────────────────
                is_train = phase.inject_feedback
                prompt_style = resolve_prompt_style(style_cfg, is_train, rng_runtime)

                # ── build prompt (mission text only — prologue already sent) ──
                if phase.mission_text_builder is not None:
                    prompt_text = phase.mission_text_builder(concrete_row, prompt_style)
                else:
                    prompt_text = build_mission_text(
                        row=concrete_row,
                        prompt_style=prompt_style,
                    )

                # ── v0.8.0: combine pending feedback with mission prompt ──
                if _pending_feedback:
                    actual_prompt = _pending_feedback + "\n\n=== NEXT MISSION ===\n\n" + prompt_text
                    _pending_feedback = ""
                else:
                    actual_prompt = prompt_text

                # ── call LLM ─────────────────────────────────────────
                raw_response = ""
                parse_result = ParseResult([], False, False, "uninitialized", "")
                try:
                    structured = llm.prompt(actual_prompt, schema=ActionPlan)
                    actions = [str(t).strip().upper() for t in structured.actions]
                    if not actions:
                        raise ValueError("Schema response had empty actions.")
                    raw_response = json.dumps({"actions": actions})
                    parse_result = ParseResult(actions, True, True, "schema", "")
                except Exception as schema_err:
                    if _is_api_fail(schema_err):
                        raise APIFailError(
                            f"mechanical API failure (schema call, phase={phase.name}): {schema_err}"
                        ) from schema_err
                    try:
                        raw_response = llm.prompt(actual_prompt)
                        parse_result = parse_actions_from_raw_text(
                            raw_response, allowed_tokens, phase_action_regex,
                        )
                        if not parse_result.parse_ok:
                            parse_result.parser_error = f"schema: {schema_err} | {parse_result.parser_error}"
                    except Exception as fb_err:
                        if _is_api_fail(fb_err):
                            raise APIFailError(
                                f"mechanical API failure (raw fallback, phase={phase.name}): {fb_err}"
                            ) from fb_err
                        parse_result = ParseResult(
                            [], False, False, "llm_call_failed",
                            f"{schema_err} | {fb_err}",
                        )

                # ── score against gold ────────────────────────────────
                gold_actions = build_gold_plan_for_row(concrete_row, phase_env, rules=phase.rules)
                target_cond = concrete_row[f"{concrete_row['target_zone']}_condition"]
                mission_result = evaluate_actions_against_gold(
                    parse_result.actions, gold_actions, target_cond,
                )
                if not parse_result.parse_ok:
                    mission_result = MissionResult(
                        False, False, 0.0, "FORMAT_ERROR", 0, max(1, len(gold_actions)),
                    )

                # ── feedback ─────────────────────────────────────────
                # v0.8.0: store feedback to combine with next mission
                # prompt, eliminating wasted model acknowledgment.
                feedback_ack = ""
                feedback_payload = ""
                if is_train and fb_mode != "none":
                    _fb_prefix = (
                        "Correct." if mission_result.mission_success else "Incorrect."
                    )
                    _fb_body = (
                        f" The optimal action sequence for that "
                        f"mission was: {json.dumps(gold_actions)}"
                    )
                    feedback_payload = _fb_prefix + _fb_body
                    _pending_feedback = feedback_payload

                # ── build trace row ──────────────────────────────────
                trace: dict[str, Any] = {
                    "task_id": row.get("task_id", task_id),
                    "mission_id": row.get("mission_id", f"{phase.name}_{mission_idx}"),
                    "source_mission_id": row.get("source_mission_id", ""),
                    "template_id": row.get("template_id", ""),
                    "goal_level": int(row.get("goal_level", 0)),
                    "target_zone": row.get("target_zone", ""),
                    "test_category": str(row.get("test_category", "") or ""),
                    "_scarcity_affected": bool(row.get("_scarcity_affected", False)),
                    "mission_split": row.get("mission_split", phase.stage_label),
                    "split_index": int(row.get("split_index", mission_idx)),
                    "phase_name": phase.name,
                    "stage": phase.stage_label,
                    "prompt_style": prompt_style,
                    "context_init_option": ctx_option,
                    "context_init_label": CONTEXT_INIT_LABELS[ctx_option],
                    "prompt_style_config": style_cfg,
                    "feedback_injected": is_train and fb_mode != "none",
                    "feedback_mode": fb_mode,
                    "feedback_ack": feedback_ack,
                    "active_rule": (
                        "scarcity" if (phase.rules and phase.rules.scarcity)
                        else "integrate" if (phase.rules and phase.rules.integrate_after_input)
                        else "log" if (phase.rules and phase.rules.log_after_management)
                        else "standard"
                    ),
                    "mission_success": mission_result.mission_success,
                    "plan_complete": mission_result.plan_complete,
                    "step_validity": mission_result.step_validity,
                    "format_success": parse_result.format_strict_ok,
                    "failure_label": mission_result.failure_label,
                    "first_error_position": mission_result.first_error_position,
                    "optimal_plan_length": mission_result.optimal_plan_length,
                    "parse_ok": parse_result.parse_ok,
                    "parse_mode": parse_result.parse_mode,
                    "parser_error": parse_result.parser_error,
                    "predicted_actions": parse_result.actions,
                    "optimal_actions": gold_actions,
                    "prompt_text": prompt_text,
                    "feedback_payload": feedback_payload,
                    "raw_response": raw_response,
                    "run_position": global_position,
                }
                for zone in phase_zones:
                    trace[f"{zone}_condition"] = concrete_row[f"{zone}_condition"]
                    trace[f"sampled_{zone}_level"] = sampled_levels.get(
                        f"sampled_{zone}_level", 0,
                    )
                phase_trace_rows.append(trace)
                all_trace_rows.append(trace)
                _recent_success.append(bool(mission_result.mission_success))

                # ── live log ─────────────────────────────────────────
                if enable_live_mission_log:
                    s = "T" if mission_result.mission_success else "F"
                    template_id = row.get("template_id", "")
                    phase_missions_so_far = mission_idx + 1
                    phase_success_so_far = sum(
                        1 for t in phase_trace_rows if t.get("mission_success", False)
                    )
                    _cond_lbl = session_label or ctx_option
                    print(
                        f"{_cond_lbl} [{phase.name}] {row.get('mission_id', '')} | "
                        f"{template_id} | "
                        f"prompt-style={prompt_style[0]} | "
                        f"success={s} | "
                        f"phase_acc=({phase_success_so_far}/{phase_missions_so_far})"
                    )

            # ── flush pending feedback from last training mission ───
            if _pending_feedback:
                try:
                    llm.prompt(_pending_feedback)
                except Exception:
                    pass
                _pending_feedback = ""

            # ── phase summary log ────────────────────────────────────
            if enable_live_mission_log and phase_trace_rows:
                _cond_lbl = session_label or ctx_option
                _total = len(phase_trace_rows)
                _correct = sum(1 for t in phase_trace_rows if t.get("mission_success", False))
                _acc = _correct / _total if _total else 0.0
                print(
                    f"  ▸ {_cond_lbl} [{phase.name}] done — "
                    f"acc={_acc:.3f} ({_correct}/{_total})"
                )

            # ── store phase traces ───────────────────────────────────
            if phase_trace_rows:
                pdf = pd.DataFrame(phase_trace_rows)
                phase_traces[phase.name] = pdf
                phase_metrics[phase.name] = summarize_learning_metrics(
                    pdf,
                    task_id=task_id,
                    prompt_config_key=ctx_option,
                    prompt_config_label=CONTEXT_INIT_LABELS[ctx_option],
                    prompt_config_depths=[],
                    context_init_style=ctx_option,
                )

            # ── retention probes after this phase ────────────────────
            if phase.name in retention_after and saved_train_missions:
                retention_rows: list[dict[str, Any]] = []
                for ret_idx, saved_row in enumerate(saved_train_missions):
                    global_position += 1
                    prompt_style_ret = resolve_prompt_style(style_cfg, False, rng_runtime)
                    prompt_text_ret = build_mission_text(
                        row=saved_row,
                        prompt_style=prompt_style_ret,
                    )
                    raw_ret = ""
                    pr_ret = ParseResult([], False, False, "uninitialized", "")
                    try:
                        structured_ret = llm.prompt(prompt_text_ret, schema=ActionPlan)
                        act_ret = [str(t).strip().upper() for t in structured_ret.actions]
                        if not act_ret:
                            raise ValueError("Empty")
                        raw_ret = json.dumps({"actions": act_ret})
                        pr_ret = ParseResult(act_ret, True, True, "schema", "")
                    except Exception:
                        try:
                            raw_ret = llm.prompt(prompt_text_ret)
                            pr_ret = parse_actions_from_raw_text(
                                raw_ret, base_allowed_tokens, canonical_action_regex,
                            )
                        except Exception:
                            pr_ret = ParseResult([], False, False, "llm_call_failed", "")

                    gold_ret = build_gold_plan_for_row(saved_row, env_runtime, rules=None)
                    mr_ret = evaluate_actions_against_gold(
                        pr_ret.actions, gold_ret,
                        saved_row[f"{saved_row['target_zone']}_condition"],
                    )
                    if not pr_ret.parse_ok:
                        mr_ret = MissionResult(False, False, 0.0, "FORMAT_ERROR", 0, max(1, len(gold_ret)))

                    ret_trace: dict[str, Any] = {
                        "mission_id": saved_row.get("mission_id", f"ret_{ret_idx}"),
                        "phase_name": f"retention_after_{phase.name}",
                        "stage": "retention",
                        "goal_level": int(saved_row.get("goal_level", 0)),
                        "target_zone": saved_row.get("target_zone", ""),
                        "active_rule": "standard",
                        "mission_success": mr_ret.mission_success,
                        "plan_complete": mr_ret.plan_complete,
                        "step_validity": mr_ret.step_validity,
                        "failure_label": mr_ret.failure_label,
                        "predicted_actions": pr_ret.actions,
                        "optimal_actions": gold_ret,
                        "run_position": global_position,
                    }
                    retention_rows.append(ret_trace)
                    all_trace_rows.append(ret_trace)

                    if enable_live_mission_log:
                        s = "T" if mr_ret.mission_success else "F"
                        _cond_lbl = session_label or ctx_option
                        print(f"{_cond_lbl} [retention_after_{phase.name}] {saved_row.get('mission_id','')} | success={s}")

                retention_traces[f"after_{phase.name}"] = pd.DataFrame(retention_rows)

            # ── structured quiz after this phase ─────────────────────
            if phase.name in quiz_after and structured_quiz_questions:
                q_results = run_structured_quiz(llm, structured_quiz_questions)
                q_summary = summarize_quiz_results(q_results)
                quiz_results[f"after_{phase.name}"] = {
                    "raw": q_results,
                    "summary": q_summary,
                }
                if enable_live_mission_log:
                    print(
                        f"[QUIZ after {phase.name}] "
                        f"{q_summary['total_correct']}/{q_summary['total_questions']} "
                        f"({q_summary['overall_accuracy']:.1%})"
                    )

    # ── assemble results ─────────────────────────────────────────────────
    all_traces_df = pd.DataFrame(all_trace_rows) if all_trace_rows else pd.DataFrame()
    return {
        "phase_traces": phase_traces,
        "all_traces": all_traces_df,
        "phase_metrics": phase_metrics,
        "retention_traces": retention_traces,
        "quiz_results": quiz_results,
        "saved_train_missions": saved_train_missions,
    }
