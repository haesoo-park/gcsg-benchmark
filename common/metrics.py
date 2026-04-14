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

# v0.8.0: Simple-LE = learning / ceiling.  Control condition dropped (always 0).
# Ceiling gate raised from 0.30 → 0.50.
MIN_CEILING_ACCURACY = 0.50


def compute_learning_efficiency(
    *,
    ceiling_accuracy: float,
    learning_accuracy: float,
    min_ceiling_accuracy: float = MIN_CEILING_ACCURACY,
) -> dict[str, float]:
    """Simple-LE = learning / ceiling.  Control condition removed (always 0).

    Gate: ceiling >= min_ceiling_accuracy (0.50).
    Clamp: [0, 1].
    """
    base = {
        "ceiling_accuracy": ceiling_accuracy,
        "learning_accuracy": learning_accuracy,
    }

    if ceiling_accuracy < min_ceiling_accuracy:
        return {**base, "learning_efficiency": 0.0,
                "ceiling_valid": False,
                "ceiling_fail_reason": f"ceiling {ceiling_accuracy:.3f} < min {min_ceiling_accuracy:.2f}"}

    if ceiling_accuracy <= 0:
        return {**base, "learning_efficiency": 0.0,
                "ceiling_valid": False,
                "ceiling_fail_reason": "ceiling_accuracy <= 0"}

    efficiency = max(0.0, min(1.0, learning_accuracy / ceiling_accuracy))
    return {**base, "learning_efficiency": efficiency,
            "ceiling_valid": True, "ceiling_fail_reason": None}


# ─── T10 Plasticity ─────────────────────────────────────────────────────

def compute_plasticity(
    *,
    affected_accuracy: float,
    not_affected_accuracy: float,
) -> dict[str, float]:
    """T10 plasticity = min(1.0, affected / not_affected).

    Returns dict with plasticity score, raw ratio, and component accuracies.
    """
    if not_affected_accuracy <= 0:
        return {"plasticity": 0.0, "plasticity_raw": 0.0,
                "affected_accuracy": affected_accuracy,
                "not_affected_accuracy": not_affected_accuracy,
                "valid": False, "fail_reason": "not_affected_accuracy <= 0"}

    raw = affected_accuracy / not_affected_accuracy
    clamped = max(0.0, min(1.0, raw))
    return {"plasticity": clamped, "plasticity_raw": raw,
            "affected_accuracy": affected_accuracy,
            "not_affected_accuracy": not_affected_accuracy,
            "valid": True, "fail_reason": None}


# ─── T11 Stability ──────────────────────────────────────────────────────

def compute_stability(
    *,
    retention_accuracy: float,
    main_study_accuracy: float,
) -> dict[str, float]:
    """T11 stability = min(1.0, retention / main_study).

    Returns dict with stability score, raw ratio, and component accuracies.
    """
    if main_study_accuracy <= 0:
        return {"stability": 0.0, "stability_raw": 0.0,
                "retention_accuracy": retention_accuracy,
                "main_study_accuracy": main_study_accuracy,
                "valid": False, "fail_reason": "main_study_accuracy <= 0"}

    raw = retention_accuracy / main_study_accuracy
    clamped = max(0.0, min(1.0, raw))
    return {"stability": clamped, "stability_raw": raw,
            "retention_accuracy": retention_accuracy,
            "main_study_accuracy": main_study_accuracy,
            "valid": True, "fail_reason": None}


# ─── Per-level LE decomposition (secondary metric) ──────────────────────

def compute_per_level_le(
    *,
    ceiling_traces: pd.DataFrame,
    learning_traces: pd.DataFrame,
    min_ceiling_accuracy: float = MIN_CEILING_ACCURACY,
) -> dict[str, float | None]:
    """Compute Simple-LE per goal level (L1, L2, L3)."""
    results: dict[str, float | None] = {}
    for level in [1, 2, 3]:
        ceil_level = ceiling_traces[ceiling_traces["goal_level"] == level]
        learn_level = learning_traces[learning_traces["goal_level"] == level]

        if len(ceil_level) == 0 or len(learn_level) == 0:
            results[f"le_L{level}"] = None
            continue

        ceil_acc = float(ceil_level["mission_success"].mean())
        learn_acc = float(learn_level["mission_success"].mean())

        le_result = compute_learning_efficiency(
            ceiling_accuracy=ceil_acc, learning_accuracy=learn_acc,
            min_ceiling_accuracy=min_ceiling_accuracy,
        )
        results[f"le_L{level}"] = le_result["learning_efficiency"]
        results[f"ceiling_L{level}"] = ceil_acc
        results[f"learning_L{level}"] = learn_acc

    return results


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
