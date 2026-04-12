from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _is_kaggle() -> bool:
    return os.path.exists("/kaggle/working")


def save_run_artifacts(
    *,
    run_dir: str | Path,
    config: dict[str, Any],
    all_results: dict[str, dict[str, Any]],
    pool_summary: dict[str, Any] | None = None,
    learning_efficiency: dict[str, Any] | None = None,
) -> Path:
    """Save all experiment artifacts to *run_dir*.

    Structure::

        run_dir/
            config.json
            pool_summary.json          (if provided)
            learning_efficiency.json   (if provided)
            cross_summary.json
            {model_alias}/
                {condition_label}/
                    traces_summary.csv
                    traces_full.jsonl
                    phase_metrics.json
                    retention_{key}.csv
                    quiz_{key}.json

    Returns:
        Path to the created run directory.
    """
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    # ── Config snapshot ──────────────────────────────────────────────
    _write_json(run_path / "config.json", config)

    # ── Pool summary ─────────────────────────────────────────────────
    if pool_summary is not None:
        _write_json(run_path / "pool_summary.json", pool_summary)

    # ── Learning efficiency ──────────────────────────────────────────
    if learning_efficiency is not None:
        _write_json(run_path / "learning_efficiency.json", learning_efficiency)

    # ── Per-model, per-condition ─────────────────────────────────────
    for model_alias, conditions in all_results.items():
        for cond_label, result in conditions.items():
            cond_path = run_path / model_alias / cond_label
            cond_path.mkdir(parents=True, exist_ok=True)

            # Traces
            all_traces: pd.DataFrame = result.get("all_traces", pd.DataFrame())
            if not all_traces.empty:
                heavy_cols = {
                    "prompt_text", "feedback_payload", "raw_response",
                    "predicted_actions", "optimal_actions", "feedback_ack",
                }
                summary_cols = [c for c in all_traces.columns if c not in heavy_cols]
                all_traces[summary_cols].to_csv(
                    cond_path / "traces_summary.csv", index=False,
                )
                all_traces.to_json(
                    cond_path / "traces_full.jsonl",
                    orient="records", lines=True,
                )

            # Phase metrics
            phase_metrics = result.get("phase_metrics", {})
            if phase_metrics:
                _write_json(cond_path / "phase_metrics.json", phase_metrics)

            # Retention traces
            for ret_key, ret_df in result.get("retention_traces", {}).items():
                if isinstance(ret_df, pd.DataFrame) and len(ret_df):
                    ret_df.to_csv(cond_path / f"retention_{ret_key}.csv", index=False)

            # Quiz results
            for quiz_key, quiz_data in result.get("quiz_results", {}).items():
                _write_json(cond_path / f"quiz_{quiz_key}.json", quiz_data)

    # ── Cross-condition summary ──────────────────────────────────────
    cross_summary: dict[str, Any] = {}
    for model_alias, conditions in all_results.items():
        model_summary: dict[str, Any] = {}
        for cond_label, result in conditions.items():
            pm = result.get("phase_metrics", {})
            model_summary[cond_label] = {
                phase_name: {
                    "accuracy": m.get("final_probe_accuracy", m.get("final_train_accuracy")),
                    "plan_complete": m.get(
                        "final_probe_plan_complete_rate",
                        m.get("final_train_plan_complete_rate"),
                    ),
                    "step_validity": m.get(
                        "avg_step_validity_probe",
                        m.get("avg_step_validity_train"),
                    ),
                }
                for phase_name, m in pm.items()
            }
        cross_summary[model_alias] = model_summary
    _write_json(run_path / "cross_summary.json", cross_summary)

    return run_path


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Kaggle dataset persistence
# ---------------------------------------------------------------------------

def push_artifacts_to_kaggle_dataset(
    run_dir: str | Path,
    *,
    dataset_slug: str,
    version_message: str | None = None,
) -> bool:
    """Push a run directory to a Kaggle dataset for long-term persistence.

    On Kaggle, ``/kaggle/working/`` is ephemeral — files disappear between
    sessions unless the notebook is committed (which preserves output as a
    versioned dataset).  This function provides an *explicit* persistence
    path by uploading the run artifacts to a user-owned Kaggle dataset.

    Prerequisites:
      - ``kaggle`` CLI installed and authenticated (``~/.kaggle/kaggle.json``).
      - The target dataset must already exist.  Create it once via::

            kaggle datasets create -p /path/to/initial/metadata/

        with a ``dataset-metadata.json`` containing your username/slug.

    Args:
        run_dir: Path to the saved run directory.
        dataset_slug: Kaggle dataset slug, e.g. ``"hesuoop/gsrcg-artifacts"``.
        version_message: Optional version note (defaults to timestamp).

    Returns:
        True if the push succeeded, False otherwise.
    """
    if not _is_kaggle():
        print("[artifacts] Not running on Kaggle — skipping dataset push.")
        return False

    run_path = Path(run_dir)
    if not run_path.exists():
        print(f"[artifacts] Run directory not found: {run_path}")
        return False

    msg = version_message or f"run {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    # Ensure dataset-metadata.json exists in the parent directory
    metadata_path = run_path.parent / "dataset-metadata.json"
    if not metadata_path.exists():
        _write_json(metadata_path, {
            "title": dataset_slug.split("/")[-1],
            "id": dataset_slug,
            "licenses": [{"name": "CC0-1.0"}],
        })

    try:
        result = subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(run_path.parent),
             "-m", msg, "--dir-mode", "tar"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            print(f"[artifacts] Pushed to kaggle.com/datasets/{dataset_slug}")
            return True
        else:
            print(f"[artifacts] Push failed: {result.stderr.strip()}")
            return False
    except FileNotFoundError:
        print("[artifacts] kaggle CLI not found. Install with: pip install kaggle")
        return False
    except subprocess.TimeoutExpired:
        print("[artifacts] Push timed out after 300s.")
        return False


def make_run_dir(base_dir: str | Path, label: str = "") -> Path:
    """Create a timestamped run directory under *base_dir*.

    Returns ``base_dir/run_{timestamp}[_{label}]``.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"run_{ts}_{label}" if label else f"run_{ts}"
    path = Path(base_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    return path
