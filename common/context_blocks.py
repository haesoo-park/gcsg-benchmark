from __future__ import annotations

# ---------------------------------------------------------------------------
# Context Initialization Blocks — synced with G-SRCG_Benchmark-Prompts.md
# ---------------------------------------------------------------------------
# Three named blocks that can be combined via CONTEXT_INIT_OPTIONS in prompts.py.
# Text is kept faithful to the spec so prompts match exactly.
#
# IMPORTANT — theoretical separation (per new_implementations_todo.md §7, §8):
#   * ENVIRONMENT_PROMPT strictly describes the static ecosystem:
#       zone conditions, action templates (with opaque *_input slots).
#       No operational logic, no dependency graph, no *_input resolution.
#   * COMPOSITION_SYSTEM_PROMPT owns every procedural rule: primitive
#       Chain/Ring structures, the Nesting operator (including the cyclic
#       dependency ring and *_input → TOKEN resolution), and the Execution
#       Persistence Rule. `learning` models must inductively recover all
#       of these from feedback; `ceiling` models receive the full block.
# ---------------------------------------------------------------------------

BASELINE_PROMPT = (
    "In this task, you are an engineer operating a controlled four-zone "
    "ecosystem.\n"
    "\n"
    "To do so, you will issue actions from the following set. "
    "You may use any subset of these actions, in any order you determine:\n"
    "\n"
    '"FILTER", "LIGHT", "SOW", "MINERAL", "TIME", "PHOTOSYNTHESIS", '
    '"DRILL", "AIR", "REFINE", "PLANT", "WATER", "FERTILIZE"'
)

ENVIRONMENT_PROMPT = (
    "Every zone is in one of three conditions:\n"
    "- it is in ideal condition\n"
    "- it is low on resource\n"
    "- its resource is in bad quality (with a k-degree severity)\n"
    "\n"
    'To "manage" a zone is to bring the zone\u2019s condition to ideal. '
    "Doing so involves a zone-specific action template:\n"
    "- atmosphere: [biomass_input, LIGHT, PHOTOSYNTHESIS]\n"
    "- biomass: [SOW, geosphere_input, FERTILIZE]\n"
    "- geosphere: [DRILL, reservoir_input, REFINE]\n"
    "- reservoir: [TIME, FILTER, atmosphere_input]"
)

COMPOSITION_SYSTEM_PROMPT = (
    "In this task, all transitions are deterministic: same state and same "
    "action always yield the same result. All action sequencing is governed "
    "by one goal-structure composition system. This system has:\n"
    "\n"
    "1. Primitive Structures \u2014 two execution modes based on goal type:\n"
    "      1. Chain structure (zone is low on resource): execute the "
    "zone\u2019s action template once.\n"
    "      2. Ring structure (zone has bad quality resource, k-degree "
    "severity): repeat the zone\u2019s action template from the beginning "
    "as many times as needed, stopping after exactly k template items have "
    "been processed \u2014 even if this terminates mid-template.\n"
    "            - k counts the items originally inside the zone\u2019s "
    "bracketed action template only (e.g., [item1, item2, item3] counts as "
    "3 items). It does not count cycles or total emitted tokens.\n"
    "            - If an *_input slot triggers a nested subgoal expansion, "
    "emit the full subgoal sequence at that position \u2014 but the slot "
    "still counts as only one primitive step toward k. Nested expansion "
    "depth does not add extra steps to the ring counter.\n"
    "            - The ring may terminate at any point in the template when "
    "the k-th primitive step is reached.\n"
    "\n"
    "2. Compositional Operator \u2014 Nesting:\n"
    "      - The four zones form a cyclic dependency ring: "
    "atmosphere \u2192 biomass \u2192 geosphere \u2192 reservoir \u2192 "
    "atmosphere. Each zone\u2019s action template contains one `*_input` "
    "slot referencing its dependency zone.\n"
    "      - When the dependency zone is in ideal condition, its `*_input` "
    "slot resolves to the zone\u2019s primary resource token:\n"
    "            - atmosphere_input \u2192 AIR\n"
    "            - biomass_input \u2192 PLANT\n"
    "            - geosphere_input \u2192 MINERAL\n"
    "            - reservoir_input \u2192 WATER\n"
    "      - When a dependency zone is non-ideal, its management subgoal is "
    "solved immediately and its full output replaces the `*_input` "
    "placeholder in the parent template.\n"
    "      - The subgoal output replaces `*_input` directly \u2014 no "
    "additional resource token is emitted at that slot.\n"
    "      - `*_input` slots each count as exactly one primitive step "
    "toward the ring counter, regardless of whether they resolve to a "
    "single token or a full nested sequence.\n"
    "      - The final executed output must always be flattened into a "
    "single, continuous 1D sequence of action tokens (do not output nested "
    "arrays).\n"
    "\n"
    "3. Execution Persistence Rule:\n"
    "      - Once a zone has been successfully managed, it remains ideal "
    "for all subsequent dependency lookups within the same mission."
)

# ---------------------------------------------------------------------------
# EFGHI Context Blocks — Task 8 (Compositional Transfer)
# ---------------------------------------------------------------------------

EFGHI_BASELINE_PROMPT = (
    "In this task, you are an engineer operating a controlled five-zone "
    "industrial facility.\n"
    "\n"
    "To do so, you will issue actions from the following set. "
    "You may use any subset of these actions, in any order you determine:\n"
    "\n"
    '"HEAT", "FLOW", "WIRE", "DATA", "FLUID", "IGNITE", "SMELT", "TEMPER", '
    '"PUMP", "PRESSURIZE", "ROUTE", "INSULATE", "SEAL", "CATALOG", "DRAIN", '
    '"CYCLE"'
)

EFGHI_ENVIRONMENT_PROMPT = (
    "Every zone is in one of three conditions:\n"
    "- it is in ideal condition\n"
    "- it is low on resource\n"
    "- its resource is in bad quality (with a k-degree severity)\n"
    "\n"
    'To "manage" a zone is to bring the zone\u2019s condition to ideal. '
    "Doing so involves a zone-specific action template:\n"
    "- furnace: [pipeline_input, IGNITE, SMELT, TEMPER]\n"
    "- pipeline: [PUMP, conduit_input, PRESSURIZE]\n"
    "- conduit: [ROUTE, INSULATE, archive_input, SEAL]\n"
    "- archive: [basin_input, CATALOG]\n"
    "- basin: [DRAIN, furnace_input, CYCLE]\n"
    "\n"
    "When a zone is in ideal condition, its primary resource token is:\n"
    "- furnace \u2192 HEAT\n"
    "- pipeline \u2192 FLOW\n"
    "- conduit \u2192 WIRE\n"
    "- archive \u2192 DATA\n"
    "- basin \u2192 FLUID"
)

EFGHI_COMPOSITION_SYSTEM_PROMPT = (
    "In this task, all transitions are deterministic: same state and same "
    "action always yield the same result. All action sequencing is governed "
    "by one goal-structure composition system. This system has:\n"
    "\n"
    "1. Primitive Structures \u2014 two execution modes based on goal type:\n"
    "      1. Chain structure (zone is low on resource): execute the "
    "zone\u2019s action template once.\n"
    "      2. Ring structure (zone has bad quality resource, k-degree "
    "severity): repeat the zone\u2019s action template from the beginning "
    "as many times as needed, stopping after exactly k template items have "
    "been processed \u2014 even if this terminates mid-template.\n"
    "            - k counts the items originally inside the zone\u2019s "
    "bracketed action template only (e.g., [item1, item2, item3] counts as "
    "3 items). It does not count cycles or total emitted tokens.\n"
    "            - If an *_input slot triggers a nested subgoal expansion, "
    "emit the full subgoal sequence at that position \u2014 but the slot "
    "still counts as only one primitive step toward k. Nested expansion "
    "depth does not add extra steps to the ring counter.\n"
    "            - The ring may terminate at any point in the template when "
    "the k-th primitive step is reached.\n"
    "\n"
    "2. Compositional Operator \u2014 Nesting:\n"
    "      - The five zones form a cyclic dependency ring: "
    "furnace \u2192 pipeline \u2192 conduit \u2192 archive \u2192 basin \u2192 "
    "furnace. Each zone\u2019s action template contains one `*_input` "
    "slot referencing its dependency zone.\n"
    "      - When the dependency zone is in ideal condition, its `*_input` "
    "slot resolves to the zone\u2019s primary resource token:\n"
    "            - pipeline_input \u2192 FLOW\n"
    "            - conduit_input \u2192 WIRE\n"
    "            - archive_input \u2192 DATA\n"
    "            - basin_input \u2192 FLUID\n"
    "            - furnace_input \u2192 HEAT\n"
    "      - When a dependency zone is non-ideal, its management subgoal is "
    "solved immediately and its full output replaces the `*_input` "
    "placeholder in the parent template.\n"
    "      - The subgoal output replaces `*_input` directly \u2014 no "
    "additional resource token is emitted at that slot.\n"
    "      - `*_input` slots each count as exactly one primitive step "
    "toward the ring counter, regardless of whether they resolve to a "
    "single token or a full nested sequence.\n"
    "      - The final executed output must always be flattened into a "
    "single, continuous 1D sequence of action tokens (do not output nested "
    "arrays).\n"
    "\n"
    "3. Execution Persistence Rule:\n"
    "      - Once a zone has been successfully managed, it remains ideal "
    "for all subsequent dependency lookups within the same mission."
)

TRANSFER_TRANSITION_ANNOUNCEMENT = (
    "IMPORTANT UPDATE \u2014 You are now being transferred to a new "
    "ecosystem. The zones, actions, and action templates have changed."
)
# NOTE: v0.8.5 removes the previous trailing phrase
# "but the underlying management rules remain the same." The prior wording
# was a legitimate design concession for weaker models (it mechanically
# told the model to transfer its learned rule), but it inflated T9 scores
# for any model strong enough to follow the hint without needing to infer
# structural analogy. Removing the phrase makes T9 a stricter blind-transfer
# test: the model now has to recognise, on its own, that the compositional
# rules induced on ABCD might apply to EFGHI.


CONTEXT_BLOCKS = {
    "baseline": BASELINE_PROMPT,
    "environment": ENVIRONMENT_PROMPT,
    "composition_system": COMPOSITION_SYSTEM_PROMPT,
    # EFGHI blocks
    "efghi_baseline": EFGHI_BASELINE_PROMPT,
    "efghi_environment": EFGHI_ENVIRONMENT_PROMPT,
    "efghi_composition_system": EFGHI_COMPOSITION_SYSTEM_PROMPT,
    "transfer_transition": TRANSFER_TRANSITION_ANNOUNCEMENT,
}
