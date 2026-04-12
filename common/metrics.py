from __future__ import annotations

from typing import Any

import pandas as pd


def missions_to_threshold(success_values: list[float], threshold: float = 0.75) -> int:
    cumulative = 0.0
    for index, value in enumerate(success_values, start=1):
        cumulative += float(value)
        if cumulative / index >= threshold:
            return index
    return -1


def summarize_learning_metrics(
    trace_df: pd.DataFrame,
    *,
    task_id: str,
    prompt_config_key: str,
    prompt_config_label: str,
    prompt_config_depths: list[int],
    context_init_style: str,
) -> dict[str, Any]:
    train_trace = trace_df[trace_df["feedback_injected"]].copy()
    probe_trace = trace_df[~trace_df["feedback_injected"]].copy()

    pre_n = min(3, len(train_trace))
    post_n = min(3, len(train_trace))
    pre_train = float(train_trace.head(pre_n)["mission_success"].mean()) if pre_n else 0.0
    post_train = float(train_trace.tail(post_n)["mission_success"].mean()) if post_n else 0.0

    train_curve = (
        train_trace.sort_values("run_position")
        .assign(train_step=lambda frame: range(1, len(frame) + 1))
        .groupby("train_step")["mission_success"]
        .mean()
        .to_dict()
    )
    probe_curve = (
        probe_trace.sort_values("run_position")
        .assign(probe_step=lambda frame: range(1, len(frame) + 1))
        .groupby("probe_step")["mission_success"]
        .mean()
        .to_dict()
    )

    metrics: dict[str, Any] = {
        "task_id": task_id,
        "prompt_config_key": prompt_config_key,
        "prompt_config_label": prompt_config_label,
        "prompt_config_depths": prompt_config_depths,
        "context_init_style": context_init_style,
        "train_mission_count": int(len(train_trace)),
        "probe_mission_count": int(len(probe_trace)),
        # --- Primary metric: exact match with optimal gold sequence (mission_success = exact_match) ---
        "pre_train_accuracy": pre_train,
        "post_train_accuracy": post_train,
        "learning_gain_train": post_train - pre_train,
        "final_train_accuracy": float(train_trace["mission_success"].mean()) if len(train_trace) else 0.0,
        "final_probe_accuracy": float(probe_trace["mission_success"].mean()) if len(probe_trace) else 0.0,
        "sample_efficiency_train_to_0_75": int(
            missions_to_threshold(train_trace["mission_success"].tolist(), threshold=0.75)
        ),
        # --- Secondary metric: plan_complete (gold prefix satisfied; extra trailing tokens allowed) ---
        "final_train_plan_complete_rate": float(train_trace["plan_complete"].mean()) if len(train_trace) else 0.0,
        "final_probe_plan_complete_rate": float(probe_trace["plan_complete"].mean()) if len(probe_trace) else 0.0,
        # Gap between plan_complete and mission_success isolates REDUNDANT_ACTIONS failure mode
        "probe_redundant_action_rate": (
            float(probe_trace["plan_complete"].mean()) - float(probe_trace["mission_success"].mean())
            if len(probe_trace) else 0.0
        ),
        # --- Partial-credit signal ---
        "avg_step_validity_train": float(train_trace["step_validity"].mean()) if len(train_trace) else 0.0,
        "avg_step_validity_probe": float(probe_trace["step_validity"].mean()) if len(probe_trace) else 0.0,
        # --- Format and parse ---
        "format_success_rate": float(trace_df["format_success"].mean()) if len(trace_df) else 0.0,
        # --- Curves and diagnostics ---
        "train_accuracy_curve": train_curve,
        "probe_accuracy_curve": probe_curve,
        "failure_counts": trace_df["failure_label"].value_counts(dropna=False).to_dict(),
        "parse_mode_counts": trace_df["parse_mode"].value_counts(dropna=False).to_dict(),
    }
    return metrics


# ─── Cross-condition learning efficiency ─────────────────────────────────

# Comprehension Threshold (new_implementations_todo.md §4). Dropped from 0.50
# to 0.30 so marginally-solvable conditions still earn partial credit. Ceilings
# below 0.30 still short-circuit LE to 0.0 (the task is considered too hard to
# confirm comprehension).
MIN_CEILING_ACCURACY = 0.30


def compute_learning_efficiency(
    *,
    ceiling_accuracy: float,
    learning_accuracy: float,
    control_accuracy: float,
    min_ceiling_accuracy: float = MIN_CEILING_ACCURACY,
) -> dict[str, float]:
    """Compute learning efficiency from three experimental conditions.

    learning_efficiency = (learning - control) / (ceiling - control)

    Conditions:
        ceiling:  Full context + feedback   (what's possible with full rules)
        learning: Baseline only + feedback   (what the model learns from feedback)
        control:  Baseline only + NO feedback (pretraining bias / exposure)

    Validity gates:
        1. ceiling_accuracy >= min_ceiling_accuracy  — ceiling must demonstrate
           the task is meaningfully solvable. A ceiling of 0.3 and learning of
           0.2 would otherwise score LE=0.67, equal to ceiling=1.0/learning=0.67.
           If this fails, learning_efficiency = 0.0.
        2. ceiling_accuracy > control_accuracy  — positive denominator.
           If this fails, learning_efficiency = 0.0.
        3. If learning_accuracy > ceiling_accuracy, the model exceeded the
           ceiling, so learning_efficiency is clamped to 1.0 (perfect score).

    Returns:
        Dict with raw values, learning_efficiency (clamped to [0,1] when valid),
        ceiling_valid (bool), and ceiling_fail_reason (str or None).
    """
    base = {
        "ceiling_accuracy": ceiling_accuracy,
        "learning_accuracy": learning_accuracy,
        "control_accuracy": control_accuracy,
        "raw_learning_gain": learning_accuracy - control_accuracy,
        "ceiling_headroom": ceiling_accuracy - control_accuracy,
    }

    # Gate 1: ceiling must be high enough to demonstrate task solvability.
    if ceiling_accuracy < min_ceiling_accuracy:
        return {**base, "learning_efficiency": 0.0,
                "ceiling_valid": False,
                "ceiling_fail_reason": f"ceiling_accuracy {ceiling_accuracy:.3f} < min {min_ceiling_accuracy:.2f}"}

    # Gate 2: positive denominator.
    denominator = ceiling_accuracy - control_accuracy
    if denominator <= 0:
        return {**base, "learning_efficiency": 0.0,
                "ceiling_valid": False,
                "ceiling_fail_reason": "ceiling_accuracy <= control_accuracy"}

    # If learning exceeds ceiling, clamp to 1.0 (model exceeded the ceiling).
    efficiency = max(0.0, min(1.0, (learning_accuracy - control_accuracy) / denominator))
    return {**base, "learning_efficiency": efficiency,
            "ceiling_valid": True, "ceiling_fail_reason": None}


def summarize_phase_comparison(
    phase_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare metrics across phases for adaptation and retention analysis.

    Expects phase_metrics dict with keys like 'original_probe', 'adapted_probe',
    'retention_post_reversal'. Computes deltas and adaptation indicators.
    """
    summary: dict[str, Any] = {"phases_present": list(phase_metrics.keys())}

    def _safe_acc(phase_name: str) -> float | None:
        m = phase_metrics.get(phase_name)
        if m is None:
            return None
        return float(m.get("final_probe_accuracy", m.get("final_train_accuracy", 0.0)))

    original_acc = _safe_acc("original_probe")
    adapted_acc = _safe_acc("adapted_probe")
    retention_acc = _safe_acc("retention_post_reversal")

    if original_acc is not None:
        summary["original_probe_accuracy"] = original_acc
    if adapted_acc is not None:
        summary["adapted_probe_accuracy"] = adapted_acc
        if original_acc is not None:
            summary["adaptation_delta"] = adapted_acc - original_acc
    if retention_acc is not None:
        summary["retention_accuracy"] = retention_acc
        if original_acc is not None:
            summary["retention_delta"] = retention_acc - original_acc
            summary["interference"] = original_acc - retention_acc

    return summary
