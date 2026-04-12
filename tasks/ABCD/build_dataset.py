from __future__ import annotations

import csv
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASK_MATRIX_PATH = ROOT / "task_matrix.json"
OUTPUT_SEQUENCE_PATH = ROOT / "dataset_sequence_complete.csv"
OUTPUT_STATE_PATH = ROOT / "dataset_state_complete.csv"

ZONES = ["atmosphere", "biomass", "geosphere", "reservoir"]
DEPENDENCY_BY_ZONE = {
    "atmosphere": "biomass",
    "biomass": "geosphere",
    "geosphere": "reservoir",
    "reservoir": "atmosphere",
}
NON_IDEAL_CONDITIONS = ["low_resource", "bad_quality"]
ALL_CONDITIONS = ["ideal", "low_resource", "bad_quality"]
INPUT_ACTION_BY_ZONE = {
    "atmosphere": "AIR",
    "biomass": "PLANT",
    "geosphere": "MINERAL",
    "reservoir": "WATER",
}
PRIMITIVE_TEMPLATE_BY_ZONE = {
    "atmosphere": ["BIOMASS_INPUT", "LIGHT", "PHOTOSYNTHESIS"],
    "biomass": ["SOW", "GEOSPHERE_INPUT", "FERTILIZE"],
    "geosphere": ["DRILL", "RESERVOIR_INPUT", "REFINE"],
    "reservoir": ["TIME", "FILTER", "ATMOSPHERE_INPUT"],
}


def build_dependency_chain(target_zone: str, depth: int) -> list[str]:
    chain = [target_zone]
    while len(chain) < depth:
        chain.append(DEPENDENCY_BY_ZONE[chain[-1]])
    return chain


def symbolic_plan(zone: str, zone_conditions: dict[str, str]) -> str:
    condition = zone_conditions[zone]
    if condition == "ideal":
        raise ValueError("symbolic_plan should only be called on non-ideal zones.")
    op = "C" if condition == "low_resource" else "R"
    emitted: list[str] = []
    for step in PRIMITIVE_TEMPLATE_BY_ZONE[zone]:
        if step.endswith("_INPUT"):
            dep_zone = DEPENDENCY_BY_ZONE[zone]
            if zone_conditions[dep_zone] == "ideal":
                emitted.append(INPUT_ACTION_BY_ZONE[dep_zone])
            else:
                emitted.append(symbolic_plan(dep_zone, zone_conditions))
        else:
            emitted.append(step)
    return f"{op}[{', '.join(emitted)}]"


def template_suffix(active_pattern: tuple[str, ...]) -> str:
    return "".join("L" if token == "low_resource" else "B" for token in active_pattern)


def generate_symbolic_rows(mode: str) -> list[dict[str, str]]:
    if mode not in {"sequence", "state"}:
        raise ValueError(f"Unsupported mode: {mode}")
    rows: list[dict[str, str]] = []
    synthetic_id = 1
    max_goal_level = len(ZONES) - 1
    for goal_level in range(1, max_goal_level + 1):
        for target_zone in ZONES:
            chain = build_dependency_chain(target_zone, goal_level + 1)
            active_chain = chain[:goal_level]
            anchor_zone = chain[goal_level]
            free_zones = [zone for zone in ZONES if zone not in active_chain and zone != anchor_zone]
            for active_pattern in itertools.product(NON_IDEAL_CONDITIONS, repeat=goal_level):
                base_conditions = {zone: "ideal" for zone in ZONES}
                for idx, zone in enumerate(active_chain):
                    base_conditions[zone] = active_pattern[idx]

                free_assignments = [None] if mode == "sequence" else list(
                    itertools.product(ALL_CONDITIONS, repeat=len(free_zones))
                )
                for free_assignment in free_assignments:
                    zone_conditions = dict(base_conditions)
                    if free_assignment is not None:
                        for idx, zone in enumerate(free_zones):
                            zone_conditions[zone] = free_assignment[idx]
                    rows.append(
                        {
                            "source_mission_id": f"abcd_{mode}_{synthetic_id:04d}",
                            "goal_level": str(goal_level),
                            "target_zone": target_zone,
                            "atmosphere_condition": zone_conditions["atmosphere"],
                            "biomass_condition": zone_conditions["biomass"],
                            "geosphere_condition": zone_conditions["geosphere"],
                            "reservoir_condition": zone_conditions["reservoir"],
                            "template_id": f"L{goal_level}_{target_zone[:3].upper()}_{template_suffix(active_pattern)}",
                            "gold_action_sequence": symbolic_plan(target_zone, zone_conditions),
                        }
                    )
                    synthetic_id += 1
    return rows


def assign_splits(
    symbolic_rows: list[dict[str, str]], *, split_seed: int, train_ratio: float, train_split_name: str, test_split_name: str
) -> list[dict[str, str]]:
    rows_by_level: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in symbolic_rows:
        rows_by_level[int(row["goal_level"])].append(dict(row))

    selected_rows: list[dict[str, str]] = []
    for goal_level in sorted(rows_by_level.keys()):
        level_rows = list(rows_by_level[goal_level])
        level_rows.sort(
            key=lambda row: (
                row["target_zone"],
                row["template_id"],
                row["atmosphere_condition"],
                row["biomass_condition"],
                row["geosphere_condition"],
                row["reservoir_condition"],
                row["source_mission_id"],
            )
        )
        for index, row in enumerate(level_rows, start=1):
            row["mission_id"] = f"L{goal_level}-{index:04d}"

        rng = random.Random(split_seed + goal_level)
        rng.shuffle(level_rows)
        train_count = int(round(len(level_rows) * train_ratio))
        train_count = min(max(train_count, 1), len(level_rows) - 1) if len(level_rows) > 1 else len(level_rows)
        train_rows = level_rows[:train_count]
        test_rows = level_rows[train_count:]
        for index, row in enumerate(train_rows, start=1):
            selected_rows.append({**row, "mission_split": train_split_name, "split_index": str(index)})
        for index, row in enumerate(test_rows, start=1):
            selected_rows.append({**row, "mission_split": test_split_name, "split_index": str(index)})
    return selected_rows


def write_dataset(path: Path, rows: list[dict[str, str]], task_id: str, train_split_name: str, test_split_name: str) -> None:
    output_columns = [
        "task_id",
        "mission_id",
        "source_mission_id",
        "goal_level",
        "mission_split",
        "split_index",
        "target_zone",
        "atmosphere_condition",
        "biomass_condition",
        "geosphere_condition",
        "reservoir_condition",
        "template_id",
        "gold_action_sequence",
    ]
    split_order = {train_split_name: 0, test_split_name: 1}
    rows = [{**row, "task_id": task_id} for row in rows]
    rows.sort(key=lambda row: (int(row["goal_level"]), split_order[row["mission_split"]], int(row["split_index"])))

    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=output_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in output_columns})
    print(f"Wrote dataset: {path}")
    print(f"Rows: {len(rows)}")


def main() -> None:
    task = json.loads(TASK_MATRIX_PATH.read_text(encoding="utf-8"))["tasks"][0]
    split_seed = int(task["seeds"]["split_seed"])
    train_ratio = float(task["protocol"]["train_ratio_per_level"])
    train_split_name = task["protocol"]["train_split_name"]
    test_split_name = task["protocol"]["test_split_name"]

    sequence_rows = assign_splits(
        generate_symbolic_rows("sequence"),
        split_seed=split_seed,
        train_ratio=train_ratio,
        train_split_name=train_split_name,
        test_split_name=test_split_name,
    )
    state_rows = assign_splits(
        generate_symbolic_rows("state"),
        split_seed=split_seed,
        train_ratio=train_ratio,
        train_split_name=train_split_name,
        test_split_name=test_split_name,
    )
    write_dataset(OUTPUT_SEQUENCE_PATH, sequence_rows, task["task_id"], train_split_name, test_split_name)
    write_dataset(OUTPUT_STATE_PATH, state_rows, task["task_id"], train_split_name, test_split_name)


if __name__ == "__main__":
    main()
