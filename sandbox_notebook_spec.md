# GCSG Sandbox Notebook — Build Specification

> **Purpose of this file.** This is a self-contained build prompt for an AI agent (or human collaborator) who will implement the GCSG sandbox notebook. It assumes the agent will first read `agent_context_prompt.md` for project background and `agi-benchmark/G-SRCG_Benchmark_Plan_v2.md` for task design. This document tells the agent *what to build and why*, not *how the underlying grammar works*.

> **Scope boundary.** This spec describes a v1 sandbox notebook sufficient to ship with the Kaggle benchmark page. It is NOT a full framework redesign. Out-of-scope extensions are enumerated in §9 and §12.

---

## 0. Handoff context

You are building a **sandbox notebook** — a companion to the 11 scored-task notebooks under `agi-benchmark/notebook/task-*.ipynb`. The sandbox is not on the leaderboard. Its job is to let a user instantiate any parameterization of the GCSG framework — either one of the 11 canonical scored task configurations, or a non-canonical variant enumerated in the benchmark's §Scalability section (deeper recursion, combined holdouts, wider k-severity, sparser training, chained adaptations, etc.) — from a single configurable cell.

Before starting:
1. Read `agent_context_prompt.md` §§ 1–3 and §10 (Gotchas). The Gotchas encode design decisions you MUST preserve.
2. Skim `agi-benchmark/G-SRCG_Benchmark_Plan_v2.md` §§ Task Design, Experimental Conditions & Metric Design, and the per-task sections for T1, T3, T6, T9, T10, T11 (these cover all five split strategies and all three schedule types).
3. Read one existing scored-task notebook end-to-end — **recommend `agi-benchmark/notebook/task-1.ipynb`** — to see the canonical `@kbench.task` pattern, the install-cell contents, and the `_emit_secondaries` helper usage.
4. Read `agi-benchmark/notebook/_generate_notebooks.py` to understand how notebooks are generated (not hand-edited) and how the shared install block is constructed.

---

## 1. Goal of the sandbox (what success looks like)

A user opens `agi-benchmark/notebook/sandbox.ipynb`, sets `canonical_task_id = "t1"` in the config cell, runs it, and gets the same LE (within run-order variance) they would get running `task-1.ipynb`. That's the reproducibility guarantee.

A user opens the same notebook, sets `canonical_task_id = None` and instead dials in `max_depth = 4`, `probe_k_range = (15, 20)`, `holdout_criteria = ["target_is_biomass", "geosphere_is_bad_quality"]`, and runs it. They get an LE for a configuration that doesn't exist as a scored task. That's the extensibility guarantee.

If the user's config is in the set of scalability variants documented on the Kaggle benchmark page (§7 Scalability — deeper recursion, combined holdouts, wider k-severity, sparser training, chained adaptations, additional prompt styles), the sandbox must run it without package modifications.

---

## 2. Notebook architecture

Match the existing scored-task notebook pattern exactly. Do not invent new conventions.

**Cell 1 — Install / imports** (copy from existing notebooks' install cell):
```python
# pip install the pinned gcsg-common version + any deps
# import kaggle_benchmarks as kbench
# import gcsg_common.* modules as used by task-1
```

**Cell 2 — Config** (this is the only cell the user edits):
- All parameters from §4 below, with defaults that reproduce T1.
- Includes a `PRESETS` dict mapping `"t1".."t11"` to full parameter sets that match the 11 scored tasks.
- If `canonical_task_id` is set, the preset overrides the individual parameters. If `None`, individual parameters apply.
- Add one commented-out "recipes" block with 3–5 non-canonical presets the user can copy (e.g., `# PRESET_L4_DEPTH = {...}`, `# PRESET_COMBINED_T5_T8 = {...}`).

**Cell 3 — Task function** decorated with `@kbench.task`. Reads the config, dispatches to `run_phased_session` with the correct `gate_check` kwarg when adaptation is enabled, handles `GateFailedError`, computes the appropriate primary metric (Simple-LE, WLC-LE, plasticity, or stability), clamps to [0, 1], returns a float.

**Cell 4 — Secondary metric emission.** Reuse `_emit_secondaries` from the existing shared install block (do not reimplement). Emit per-level, per-condition, and `format_valid_rate`. Auto-enable `by_template=True` when the config is T6-like (combo holdout) or T7-like (non-canonical prompt styles) or T9-like (transfer environment).

---

## 3. Parameter surface (full list)

Config cell variables, grouped by axis. Types given in Python notation.

### 3.1 Canonical preset override
- `canonical_task_id: Optional[str]` — one of `"t1".."t11"` or `None`. When set, loads the matching entry from `PRESETS` and ignores the individual parameters below. When `None`, individual parameters apply.

### 3.2 Environment
- `environment: Literal["ABCD", "EFGHI"]` — which dataset to sample from. Default `"ABCD"`. (Multi-environment chaining for T9-like transfer is set via §3.8 `transfer_environment`.)

### 3.3 Dataset & split
- `split_type: Literal["canonical", "custom"]` — `"canonical"` reads `t{N}_split` CSV column (fast, deterministic, matches scored tasks). `"custom"` invokes the split logic fresh with the parameters below.
- `split_strategy: Literal["random-small", "goalwise", "configuration-holdout", "combo-holdout", "k-severity-holdout", "transfer-probe"]` — applies only when `split_type="custom"`.
- `train_level_counts: Dict[int, int]` — missions per level in the train set, e.g., `{1: 8, 2: 8, 3: 8}`. Keys can extend to `4` for L4 variants.
- `probe_level_counts: Dict[int, int]` — analogous.
- `holdout_criteria: List[str]` — for configuration-holdout. Accepts the same criteria names T5 uses (`"target_is_biomass"`, `"geosphere_is_bad_quality"`). Multi-element list enables combined holdouts.
- `combo_holdout_train: List[str]` — for combo-holdout, e.g., `["CC", "RR"]`.
- `combo_holdout_probe: List[str]` — e.g., `["CR", "RC"]`.
- `l1_contrastive_pairs: int` — default 3; applies to L1 train composition.
- `split_seed: int` — default 1. Preserved for ablation only; canonical path ignores.

### 3.4 K-severity
- `train_k_range: Tuple[int, int]` — default `(4, 10)`.
- `probe_k_range: Tuple[int, int]` — default `(4, 10)`. Disjoint from train for T8-style parametric extrapolation (e.g., `(11, 14)` with train `(4, 8)` creates the k=9–10 dead zone).

### 3.5 Prompt / context
- `context_init: Literal["baseline", "environment", "composition-rules"]` — what the system prompt contains. `"baseline"` = T2 learning, `"environment"` = T1 learning, `"composition-rules"` = ceiling (full rules).
- `train_prompt_style: str` — one of `"canonical"`, `"zone-reordered"`, `"word-resource-absent"`, `"prose-like"`, `"zone-name-absent"`. Default `"canonical"`.
- `probe_prompt_style: Union[str, Literal["random-non-canonical"], List[str]]` — `"canonical"` for most tasks; `"random-non-canonical"` reproduces T7. A `List[str]` rotates deterministically across probe missions.

### 3.6 Schedule
- `schedule: Literal["probe-at-end", "interleaved-blocks"]` — default `"probe-at-end"`. `"interleaved-blocks"` reproduces T3.
- `n_blocks: int` — default 8. Applies only to interleaved.
- `train_per_block: int` — default 3.
- `probe_per_block: int` — default 3.
- `block_level_balance: bool` — default `True`; each block contains 1×L1 + 1×L2 + 1×L3 (like T3).

### 3.7 Adaptation / retention
- `adaptation_rule: Optional[Literal["scarcity", "integrate", "custom"]]` — `None` for T1–T9; `"scarcity"` for T10; `"integrate"` for T11. `"custom"` is an extension point — log a warning that it's not implemented in v1.
- `adaptation_n_blocks: int` — default 5.
- `adaptation_train_per_block: int` — default 2.
- `adaptation_probe_per_block: int` — default 4.
- `adaptation_affected_ratio: float` — default 0.5 (T10: 2 affected + 2 not-affected per probe block). Set to 1.0 for T11 (all adaptation probes are rule-affected).
- `adaptation_signal: Literal["change-only", "rule-text"]` — default `"change-only"` (matches T10/T11). `"rule-text"` lets the user expose the new rule verbatim as an ablation.
- `retention_study: Optional[Literal["rule-reverse"]]` — `None` for T10; `"rule-reverse"` for T11.
- `chained_adaptation: Optional[List[dict]]` — extension point for "rule A → rule B → reverse A" sequences enumerated in §7 Scalability. In v1 sandbox, accept the parameter but document "not yet implemented" and emit a clear error. Full implementation is a follow-on.

### 3.8 Transfer
- `transfer_environment: Optional[Literal["EFGHI"]]` — `None` for T1–T8, T10–T11. `"EFGHI"` reproduces T9 (train ABCD, probe EFGHI).
- `transfer_probe_level_counts: Dict[int, int]` — for T9, current canonical is `{2: 8, 3: 16, 4: 8}` (v0.8.4 redistribution; L1 intentionally omitted, see `G-SRCG_Benchmark_Plan_v2.md` §T9 Split).
- `include_verify_probe: bool` — default `True`; adds the 8-mission unscored ABCD verification probe used by T9.

### 3.9 Feedback / training
- `train_with_feedback: bool` — default `True` for learning, `False` for ceiling.
- `probe_with_feedback: bool` — default `False`.
- `reflection_after_feedback: bool` — default `False`. Legacy; keep the knob but do NOT recommend enabling (see Gotcha #19 — reflection does not substitute for extended thinking).

### 3.10 Gates
- `comprehension_threshold: float` — default 0.50. If `ceiling_probe_acc < comprehension_threshold`, LE is forced to 0.0.
- `learning_gate_threshold: float` — default 0.40. Applies to `main_study_probe_acc` in adaptation tasks.
- `fail_fast: bool` — default `True` for adaptation tasks; `False` otherwise. When `True`, the first run failing the gate assigns task score 0.0 and halts subsequent runs (Gotcha #4).

### 3.11 Multi-run
- `n_runs: int` — default 3 (matches post-Pilot-3 calibration). Each run uses the same canonical split but a different presentation-order seed.
- `base_seed: int` — default 1. Per-run seeds are `base_seed + run_idx`.

### 3.12 Output
- `emit_secondaries: bool` — default `True`.
- `emit_per_template: bool` — default `False`; auto-set to `True` when `split_strategy == "combo-holdout"`, when `probe_prompt_style` is a list or `"random-non-canonical"`, or when `transfer_environment` is set.

---

## 4. Presets (11 canonical configurations)

Define as a module-level dict in the config cell. Each key is `"t1".."t11"`; each value is a dict mapping the parameter names above to task-specific values. Source of truth for every preset:
- `G-SRCG_Benchmark_Plan_v2.md` per-task sections (§§T1–T11).
- The canonical split columns in `gcsg-benchmark/common/data/dataset_state_complete.csv` (`t1_split`–`t11_split`) and `efghi_dataset_sequence_complete.csv` (`t9_split`).

**Validation invariant:** running `canonical_task_id = "t{N}"` with all other params at defaults must reproduce the corresponding scored task's LE on Flash (within run-order variance — i.e., within ± 2 × the Flash per-run std reported in `agent_context_prompt.md` §13 Decision Log 2026-04-14). This is the only hard correctness test.

Spot-check presets to validate at minimum: `t1`, `t3` (interleaved schedule), `t6` (combo holdout), `t9` (transfer), `t10` (fail-fast behavior).

---

## 5. Invariants that must be preserved

Do not undo any of these. They correspond to hard-won design decisions logged in `agent_context_prompt.md` §10 Gotchas.

1. **Single-session for adaptation tasks.** T10/T11 configurations must run the main study and adaptation phases in a single `run_phased_session` call with `gate_check` kwarg, not two separate sessions. (Gotchas #3 + #14.)
2. **Fail-fast on gate failures.** First run with `main_study_probe_acc < learning_gate_threshold` returns 0.0 for the task and halts remaining runs — when `fail_fast=True`. (Gotcha #4.)
3. **Canonical splits via CSV columns.** When `split_type="canonical"`, call `allocate_split(..., use_canonical=True, task_number=N)`. Do not re-generate splits at runtime. (Gotcha #5.)
4. **Primary LE clamped to [0, 1], unclamped preserved as secondary.** (Gotcha #6.)
5. **Response parser stays lenient.** 4-layer fallback (JSON strict → JSON extracted → array literal → token salvage). Do not tighten. (Gotcha #9.)
6. **`API_FAIL` excluded from accuracy denominator.** Context-window and rate-limit errors are logged but not counted as cognitive failures. (Gotcha #10.)
7. **T9 LE uses EFGHI probes only.** ABCD verification probe accuracy is secondary, not primary. (Gotcha #12.)
8. **No double-session fallback.** If you see yourself about to write a two-session control or a re-probe after gate, re-read Gotcha #14 first.

---

## 6. Required integration points

Existing API surface you will call (do not reimplement):

- `gcsg_common.task_splits.allocate_split(use_canonical, task_number, ...)` — canonical path for preset configs.
- `gcsg_common.session.run_phased_session(phases, gate_check=...)` — multi-turn conversation driver. Accepts `gate_check={"after_phase", "metric", "threshold"}` and raises `GateFailedError` with a `.partial_result` attribute.
- `gcsg_common.session.GateFailedError` — wrap the session call in a try/except to realize fail-fast logic.
- `gcsg_common.metrics.compute_simple_le(learning_acc, ceiling_acc)` → float.
- `gcsg_common.metrics.compute_wlc(block_accs, weights)` → float (T3 path).
- `gcsg_common.metrics.compute_plasticity(affected, not_affected)` → float.
- `gcsg_common.metrics.compute_stability(retention, main_study)` → float.
- `gcsg_common.prompts.*` — prompt-block assembly per `context_init` setting.
- `gcsg_common.pools.build_adaptation_pools(...)` — adaptation train/probe pool builders (T10, T11).
- `_emit_secondaries(all_run_results, task_id, by_template=...)` — shared helper already injected into the notebook install block by `_generate_notebooks.py`.

If a needed integration point does not exist (e.g., combined-holdout split strategies that stack configuration + k-severity), **stop and flag it** — do not silently implement around it. See §12 Known Risks.

---

## 7. Build script

Add a generator to `agi-benchmark/notebook/_generate_notebooks.py` named `build_sandbox_notebook()` following the pattern of `build_task_notebook(task_id)`. The generated file lands at `agi-benchmark/notebook/sandbox.ipynb`.

Do not hand-edit the generated notebook. All changes must flow through the generator script — this matches the existing pattern for task-1 through task-11.

---

## 8. Acceptance criteria

Before considering this done:

- [ ] Notebook is syntax-clean (all cells execute with no errors against a mock LLM that returns any valid JSON).
- [ ] **Preset reproducibility** — running `canonical_task_id = "t1"` on Flash 2.5 reproduces `gcsg_t1_compositional_rule_induction-run_id_Run_1_google_gemini-2.5-flash.run.json` LE within ± 2σ (σ = Flash per-run std from the N=5 pilot).
- [ ] Same check for `t3`, `t6`, `t9`, `t10`.
- [ ] **Non-canonical variant** — at least one configuration NOT in the 11 presets (suggest: L4 probe on EFGHI with ABCD training, OR T5 holdout_criteria + T8 k-holdout combined) runs end-to-end and produces a numeric LE.
- [ ] Secondary metrics emitted to stdout in the same format as existing notebooks (`SECONDARY_JSON` lines).
- [ ] No file writes to the Kaggle working directory.
- [ ] Config cell is a single cell; a reader can see the entire parameter surface without scrolling through code.

---

## 9. Non-goals (v1)

Explicitly OUT of scope for v1:

- New synthetic grammars (only ABCD and EFGHI are supported).
- N-zone environments beyond 4 and 5 (documented as future work in §12).
- Cross-lingual prompt styles (listed in §7 Scalability of the benchmark page, but deferred).
- Persisting per-run LE to the `.run.json` artifact — stdout is the current persistence path (Gotcha #16).
- Chained adaptation (`rule A → rule B → reverse A`) — accept the parameter, error-out gracefully, document as follow-on.

If a user asks for any of the above, the sandbox should raise a clear `NotImplementedError` with a pointer to this spec file.

---

## 10. Expected file layout after the build

```
agi-benchmark/
├─ notebook/
│  ├─ _generate_notebooks.py        ← add build_sandbox_notebook() here
│  ├─ sandbox.ipynb                 ← NEW, generated output
│  ├─ task-1.ipynb                  ← unchanged
│  └─ task-2..task-11.ipynb         ← unchanged
```

No changes to `gcsg-benchmark/common/*` should be required. If you find yourself modifying package code, stop and surface the blocker to the human collaborator — it probably indicates the feature belongs in a v0.9 package bump, not in the v1 sandbox.

---

## 11. Known risks / open questions

Flag each of these to the human when you encounter them. Do not silently route around.

1. **Combined holdouts may not be supported by `allocate_split`.** The current function signature accepts one holdout criterion at a time. Stacking configuration-holdout + k-severity-holdout may require either (a) a package-side change to `allocate_split`, or (b) a sandbox-side composition of sequential filters. Determine which is correct before implementing.
2. **L4 sampling on EFGHI exists; on ABCD it does not.** ABCD's 4-zone ring maxes out at L3. If a user requests `max_depth = 4` with `environment = "ABCD"`, error out with a clear message pointing to EFGHI.
3. **Canonical splits only exist for t1..t11.** Any `canonical_task_id` not in that set is an error. Do not fall back to in-memory generation for canonical paths — that would defeat the purpose of canonical splits.
4. **Per-run quota cost is unbounded.** A sandbox run at `n_runs=5` on a deep-thinking model can consume ≥ $50 of Kaggle quota. Add a cell-level warning that the sandbox has no internal cost cap and prints estimated cost before execution.
5. **`reflection_after_feedback = True` is a trap.** Keep the knob but add an inline comment: "Do not enable for comprehension-threshold investigations; see Gotcha #19."
6. **v0.9 package refactor is planned** (see Decision Log 2026-04-15 entry). If you touch package-side split logic, coordinate with the owner — do not merge changes that block v0.9.

---

## 12. Follow-on tasks (post-sandbox-v1)

Not your problem for v1, but note for continuity:

- Add a "recipes" companion markdown at `agi-benchmark/notebook/sandbox_recipes.md` with 5–10 copyable parameter settings for interesting variants.
- Add a `dry_run: bool` parameter that emits the constructed prompt + mission list without calling an LLM, for prompt-engineering inspection.
- Add a link/launch button from the Kaggle benchmark page to the sandbox.
- Support chained adaptation (§3.7 `chained_adaptation`).
- Support N-zone environments via a parameter-driven environment builder.
- Support cross-lingual prompt styles.

---

## 13. What to do if you get stuck

1. Re-read `agent_context_prompt.md` §10 Gotchas. Most "stuck" moments are Gotchas in disguise.
2. Consult `agi-benchmark/G-SRCG_Benchmark_Plan_v2.md` — it is the source of truth for task design.
3. If the source of truth and this spec disagree, the plan wins and this spec is stale. Flag it to the human.
4. Do NOT modify `kaggle-benchmarks-ci/` (upstream SDK, frozen).
5. Do NOT edit canonical CSV split columns (`t1_split..t11_split`) — those are frozen at v0.8.4.

---

*End of sandbox build specification. When this document and the canonical design plan disagree, the plan wins.*
