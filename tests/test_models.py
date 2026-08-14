"""Model validation tests.

These pin the guarantees the rest of the pipeline relies on: bad LLM output
is rejected at construction, and AgentState is buildable with no arguments.
"""

import pytest
from pydantic import ValidationError

from notion_triage_agent.models import (
    ActionItem,
    AgentState,
    Classification,
    PriorityLevel,
    Recommendation,
    TaskAnalysis,
    TaskCategory,
)


def test_agent_state_starts_empty():
    """The graph is invoked with AgentState(), so it must need no arguments."""
    state = AgentState()
    assert state.raw_tasks == []
    assert state.analyses == []
    assert state.errors == []
    assert state.recommendation is None


def test_confidence_must_be_a_probability():
    for bad in (1.7, -0.1):
        with pytest.raises(ValidationError):
            Classification(category=TaskCategory.TASK, confidence=bad, reasoning="x")


def test_unknown_category_is_rejected():
    """A hallucinated category must not reach the CLI as a raw string."""
    with pytest.raises(ValidationError):
        Classification(category="todo", confidence=0.5, reasoning="x")


def test_enums_serialize_as_plain_strings():
    """str-mixin enums keep model_dump() JSON-clean."""
    dumped = Classification(
        category=TaskCategory.REFERENCE, confidence=0.5, reasoning="x"
    ).model_dump(mode="json")
    assert dumped["category"] == "reference"


def test_nested_models_validate_from_dicts():
    """One LLM call returns the whole tree; validation must recurse into it."""
    analysis = TaskAnalysis.model_validate(
        {
            "task_id": "abc",
            "classification": {
                "category": "task",
                "confidence": 0.9,
                "reasoning": "actionable",
            },
            "action_items": [{"description": "Do the thing", "estimated_minutes": 30}],
            "priority": "high",
            "priority_reasoning": "blocking",
        }
    )
    assert isinstance(analysis.classification, Classification)
    assert analysis.priority is PriorityLevel.HIGH
    assert analysis.action_items[0].estimated_minutes == 30


def test_reference_items_may_have_no_actions():
    analysis = TaskAnalysis(
        task_id="abc",
        classification=Classification(
            category=TaskCategory.REFERENCE, confidence=0.8, reasoning="x"
        ),
        priority=PriorityLevel.LOW,
        priority_reasoning="x",
    )
    assert analysis.action_items == []


def test_estimates_may_be_absent():
    """Null discipline: the model is told to omit estimates it cannot make."""
    assert ActionItem(description="x").estimated_minutes is None
    assert Recommendation(reasoning="x").estimated_total_minutes is None
