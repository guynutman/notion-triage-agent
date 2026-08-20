"""Node tests driven by fake clients.

Nodes receive their dependencies as keyword arguments, so every test here
runs with no network, no API key, and deterministic model output. This is
the payoff of injecting the clients instead of importing them.
"""

import pytest

from notion_triage_agent import nodes
from notion_triage_agent.llm import LLMError
from notion_triage_agent.models import (
    ActionItem,
    AgentState,
    Classification,
    NotionTask,
    PriorityLevel,
    Recommendation,
    TaskAnalysis,
    TaskCategory,
)
from notion_triage_agent.notion_client import NotionAPIError


def make_task(task_id: str = "t1", title: str = "Fix auth") -> NotionTask:
    return NotionTask(id=task_id, title=title, description="d", status="Not started")


def make_analysis(task_id: str = "t1") -> TaskAnalysis:
    return TaskAnalysis(
        task_id=task_id,
        classification=Classification(category=TaskCategory.TASK, confidence=0.9, reasoning="r"),
        action_items=[ActionItem(description="do it", estimated_minutes=30)],
        priority=PriorityLevel.HIGH,
        priority_reasoning="r",
    )


class FakeNotion:
    """Satisfies the one method fetch_tasks uses."""

    def __init__(self, tasks=None, error: Exception | None = None):
        self._tasks = tasks or []
        self._error = error

    def fetch_tasks(self, filter_status=None):
        if self._error:
            raise self._error
        return self._tasks


class FakeLLM:
    """Satisfies the LLMClient protocol. Returns queued values in order.

    A queued Exception is raised instead of returned, which is how the retry
    and per-task-failure paths get exercised.
    """

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str, schema):
        self.calls.append(prompt)
        if not self._responses:
            raise LLMError("fake ran out of responses")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


# --- fetch_tasks ---------------------------------------------------------


def test_fetch_tasks_returns_only_the_field_it_changed():
    update = nodes.fetch_tasks(AgentState(), notion_client=FakeNotion([make_task()]))
    assert set(update) == {"raw_tasks"}
    assert len(update["raw_tasks"]) == 1


def test_fetch_tasks_records_api_errors_instead_of_raising():
    """An unreachable Notion should degrade, not crash the pipeline."""
    update = nodes.fetch_tasks(
        AgentState(), notion_client=FakeNotion(error=NotionAPIError("401 bad token"))
    )
    assert update["raw_tasks"] == []
    assert "401 bad token" in update["errors"][0]


# --- analyze_tasks -------------------------------------------------------


def test_analyze_tasks_handles_an_empty_database():
    update = nodes.analyze_tasks(AgentState(), llm_client=FakeLLM())
    assert update["analyses"] == []


def test_analyze_tasks_overwrites_the_model_supplied_task_id():
    """The id is derivable, so the model is never trusted to echo it."""
    state = AgentState(raw_tasks=[make_task("real-id")])
    llm = FakeLLM(make_analysis(task_id="hallucinated-id"))

    update = nodes.analyze_tasks(state, llm_client=llm)

    assert update["analyses"][0].task_id == "real-id"


def test_analyze_tasks_retries_once_before_giving_up():
    state = AgentState(raw_tasks=[make_task()])
    llm = FakeLLM(LLMError("truncated"), make_analysis())

    update = nodes.analyze_tasks(state, llm_client=llm)

    assert len(llm.calls) == 2
    assert len(update["analyses"]) == 1
    assert update["errors"] == []


def test_one_failing_task_does_not_abort_the_batch():
    state = AgentState(raw_tasks=[make_task("t1"), make_task("t2", "Other")])
    # t1 fails both attempts; t2 succeeds.
    llm = FakeLLM(LLMError("boom"), LLMError("boom"), make_analysis("t2"))

    update = nodes.analyze_tasks(state, llm_client=llm)

    assert [a.task_id for a in update["analyses"]] == ["t2"]
    assert len(update["errors"]) == 1
    assert "Fix auth" in update["errors"][0]


def test_prompt_marks_absent_fields_explicitly():
    """None must not reach the prompt as the literal string "None"."""
    state = AgentState(raw_tasks=[NotionTask(id="t1", title="Bare")])
    llm = FakeLLM(make_analysis())

    nodes.analyze_tasks(state, llm_client=llm)

    assert "(none)" in llm.calls[0] and "(unset)" in llm.calls[0]


# --- recommend -----------------------------------------------------------


def test_recommend_skips_the_model_when_there_is_nothing_to_rank():
    llm = FakeLLM()
    update = nodes.recommend(AgentState(), llm_client=llm)
    assert update["recommendation"] is None
    assert llm.calls == []


def test_recommend_drops_hallucinated_ids_and_restores_dropped_ones():
    state = AgentState(
        raw_tasks=[make_task("t1"), make_task("t2", "Other")],
        analyses=[make_analysis("t1"), make_analysis("t2")],
    )
    llm = FakeLLM(Recommendation(ranked_tasks=["t2", "does-not-exist"], reasoning="r"))

    update = nodes.recommend(state, llm_client=llm)

    # t2 keeps the model's ordering, t1 is appended, the invented id is gone.
    assert update["recommendation"].ranked_tasks == ["t2", "t1"]


def test_recommend_records_the_error_when_the_model_fails():
    state = AgentState(raw_tasks=[make_task()], analyses=[make_analysis()])
    llm = FakeLLM(LLMError("quota"), LLMError("quota"))

    update = nodes.recommend(state, llm_client=llm)

    assert update["recommendation"] is None
    assert "quota" in update["errors"][0]


# --- state contract ------------------------------------------------------


@pytest.mark.parametrize(
    "update",
    [
        nodes.fetch_tasks(AgentState(), notion_client=FakeNotion([make_task()])),
        nodes.analyze_tasks(
            AgentState(raw_tasks=[make_task()]), llm_client=FakeLLM(make_analysis())
        ),
    ],
)
def test_nodes_return_partial_updates_that_merge_into_state(update):
    """Nodes return a dict of changed fields, never a whole AgentState."""
    assert isinstance(update, dict)
    assert set(update) <= set(AgentState.model_fields)
