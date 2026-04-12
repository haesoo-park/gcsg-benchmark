from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Structured Rule Quiz — checkable questions about the environment rules
# ---------------------------------------------------------------------------

STRUCTURED_QUIZ_QUESTIONS: list[dict[str, Any]] = [
    # ── Zone Dependencies ────────────────────────────────────────────────
    {
        "id": "dep_atm",
        "question": "What zone does atmosphere depend on for its input?",
        "expected_keywords": ["biomass"],
        "category": "dependency",
    },
    {
        "id": "dep_bio",
        "question": "What zone does biomass depend on for its input?",
        "expected_keywords": ["geosphere"],
        "category": "dependency",
    },
    {
        "id": "dep_geo",
        "question": "What zone does geosphere depend on for its input?",
        "expected_keywords": ["reservoir"],
        "category": "dependency",
    },
    {
        "id": "dep_res",
        "question": "What zone does reservoir depend on for its input?",
        "expected_keywords": ["atmosphere"],
        "category": "dependency",
    },
    # ── Goal Type Rules ──────────────────────────────────────────────────
    {
        "id": "chain_rule",
        "question": (
            "When a zone is low on resource, how many times do you execute "
            "its management action template?"
        ),
        "expected_keywords": ["once", "one time", "1 time", "single"],
        "category": "goal_type",
    },
    {
        "id": "ring_rule",
        "question": (
            "When a zone has bad quality resource with severity level k, "
            "how many primitive action steps do you execute?"
        ),
        "expected_keywords": ["k", "exactly k", "k steps", "k primitive"],
        "category": "goal_type",
    },
    {
        "id": "ring_mid_template",
        "question": (
            "In a ring structure (bad quality), can the execution stop in the "
            "middle of the action template, or must it always complete a full cycle?"
        ),
        "expected_keywords": ["stop", "mid", "middle", "terminate", "partial"],
        "category": "goal_type",
    },
    # ── Input Resolution ─────────────────────────────────────────────────
    {
        "id": "input_ideal",
        "question": (
            "If biomass is in ideal condition, what single action replaces "
            "biomass_input?"
        ),
        "expected_keywords": ["plant"],
        "category": "input_resolution",
    },
    {
        "id": "input_nonideal",
        "question": (
            "If a dependency zone is NOT in ideal condition, what must you do "
            "at its *_input slot instead of the single resource action?"
        ),
        "expected_keywords": ["manage", "subgoal", "expand", "resolve", "sequence"],
        "category": "input_resolution",
    },
    # ── State Tracking ───────────────────────────────────────────────────
    {
        "id": "state_persist",
        "question": (
            "After successfully managing a dependency zone within a mission, "
            "does that zone remain in ideal condition for the rest of the mission?"
        ),
        "expected_keywords": ["yes", "remains", "persists", "stays", "ideal"],
        "category": "state",
    },
    # ── Counting ─────────────────────────────────────────────────────────
    {
        "id": "input_counts_as_step",
        "question": (
            "In a ring structure (bad quality with severity k), does an *_input "
            "slot count as one primitive step toward the k-count?"
        ),
        "expected_keywords": ["yes", "counts", "one"],
        "category": "counting",
    },
    # ── Zone Templates ───────────────────────────────────────────────────
    {
        "id": "template_atm",
        "question": (
            "What is the action template for managing the atmosphere zone? "
            "List the steps in order."
        ),
        "expected_keywords": ["plant", "light", "photosynthesis"],
        "category": "template",
    },
    {
        "id": "template_bio",
        "question": (
            "What is the action template for managing the biomass zone? "
            "List the steps in order."
        ),
        "expected_keywords": ["sow", "mineral", "fertilize"],
        "category": "template",
    },
]


def score_single_response(question: dict[str, Any], response_text: str) -> dict[str, Any]:
    """Score a single quiz response against expected keywords.

    Returns dict with: id, category, question, response, matched_keywords,
    score (1.0 if any keyword matched, else 0.0), all_keywords.
    """
    response_lower = response_text.lower().strip()
    matched = [kw for kw in question["expected_keywords"] if kw.lower() in response_lower]
    return {
        "id": question["id"],
        "category": question["category"],
        "question": question["question"],
        "response": response_text,
        "matched_keywords": matched,
        "score": 1.0 if matched else 0.0,
        "all_keywords": question["expected_keywords"],
    }


def run_structured_quiz(
    llm: Any,
    questions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Present each quiz question to the LLM and score responses.

    Args:
        llm: Language model with a .prompt() method.
        questions: Quiz questions (defaults to STRUCTURED_QUIZ_QUESTIONS).

    Returns:
        List of scored response dicts.
    """
    if questions is None:
        questions = STRUCTURED_QUIZ_QUESTIONS

    results: list[dict[str, Any]] = []
    for q in questions:
        try:
            response = str(llm.prompt(q["question"]))
        except Exception as e:
            response = f"QUIZ_ERROR: {e}"
        results.append(score_single_response(q, response))
    return results


def summarize_quiz_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate quiz results into a summary dict."""
    total = len(results)
    correct = sum(1 for r in results if r["score"] > 0)
    by_category: dict[str, dict[str, int]] = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"total": 0, "correct": 0}
        by_category[cat]["total"] += 1
        by_category[cat]["correct"] += int(r["score"] > 0)
    return {
        "total_questions": total,
        "total_correct": correct,
        "overall_accuracy": correct / max(1, total),
        "by_category": {
            cat: {**counts, "accuracy": counts["correct"] / max(1, counts["total"])}
            for cat, counts in by_category.items()
        },
        "per_question": [
            {"id": r["id"], "score": r["score"], "matched": r["matched_keywords"]}
            for r in results
        ],
    }
