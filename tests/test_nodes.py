"""Node tests driven by fake clients.

Nodes receive their dependencies as keyword arguments, so every test here
runs with no network, no API key, and deterministic model output. This is
the payoff of injecting the clients instead of importing them.
"""

import threading
import time

import pytest

from notion_triage_agent import nodes
from notion_triage_agent.llm import LLMError
from notion_triage_agent.models import (
    ActionItem,
    AgentState,
    Classification,
    DayPlan,
    NotionTask,
    PlannedTask,
    PriorityLevel,
    Recommendation,
    TaskAnalysis,
    TaskCategory,
    WeeklyPlan,
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
    """Satisfies the NotionClient methods the nodes actually call."""

    def __init__(self, tasks=None, error: Exception | None = None, write_error=None):
        self._tasks = tasks or []
        self._error = error
        self._write_error = write_error
        self.fetch_kwargs: dict = {}
        self.created_properties: list[str] = []
        self.updates: list[tuple[str, dict]] = []

    def fetch_tasks(self, filter_status=None, limit=None):
        self.fetch_kwargs = {"filter_status": filter_status, "limit": limit}
        if self._error:
            raise self._error
        return self._tasks[:limit] if limit else self._tasks

    def ensure_select_properties(self, names):
        self.created_properties = list(names)
        return list(names)

    def update_task_properties(self, task_id, properties):
        if self._write_error:
            raise self._write_error
        self.updates.append((task_id, properties))


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

    update = nodes.analyze_tasks(state, llm_client=llm, max_workers=1)

    assert len(llm.calls) == 2
    assert len(update["analyses"]) == 1
    assert update["errors"] == []


def test_one_failing_task_does_not_abort_the_batch():
    state = AgentState(raw_tasks=[make_task("t1"), make_task("t2", "Other")])
    # t1 fails both attempts; t2 succeeds.
    llm = FakeLLM(LLMError("boom"), LLMError("boom"), make_analysis("t2"))

    # max_workers=1 keeps the queue-based fake deterministic.
    update = nodes.analyze_tasks(state, llm_client=llm, max_workers=1)

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


# --- write_back ----------------------------------------------------------


def test_write_back_creates_columns_then_writes_each_task():
    notion = FakeNotion()
    state = AgentState(raw_tasks=[make_task("t1")], analyses=[make_analysis("t1")])

    update = nodes.write_back(state, notion_client=notion)

    assert notion.created_properties == [
        nodes.CATEGORY_PROPERTY,
        nodes.PRIORITY_PROPERTY,
    ]
    task_id, properties = notion.updates[0]
    assert task_id == "t1"
    assert properties[nodes.PRIORITY_PROPERTY] == {"select": {"name": "high"}}
    assert properties[nodes.CATEGORY_PROPERTY] == {"select": {"name": "task"}}
    assert update["errors"] == []


def test_write_back_records_per_page_failures():
    notion = FakeNotion(write_error=NotionAPIError("404 page not found"))
    state = AgentState(raw_tasks=[make_task("t1")], analyses=[make_analysis("t1")])

    update = nodes.write_back(state, notion_client=notion)

    assert len(update["errors"]) == 1
    assert "404" in update["errors"][0]


def test_write_back_does_nothing_without_analyses():
    notion = FakeNotion()
    assert nodes.write_back(AgentState(), notion_client=notion) == {}
    assert notion.updates == []


# --- fetch options -------------------------------------------------------


def test_fetch_tasks_passes_filter_and_limit_through():
    notion = FakeNotion([make_task("t1"), make_task("t2")])

    nodes.fetch_tasks(AgentState(), notion_client=notion, filter_status="Not started", limit=1)

    assert notion.fetch_kwargs == {"filter_status": "Not started", "limit": 1}


# --- routing -------------------------------------------------------------


def test_router_skips_analysis_on_an_empty_fetch():
    assert nodes.has_tasks(AgentState()) == "__end__"
    assert nodes.has_tasks(AgentState(raw_tasks=[make_task()])) == "analyze_tasks"


# --- parallelism ---------------------------------------------------------


def test_parallel_analysis_preserves_input_order():
    """Results are collected in submission order, not completion order."""

    class SlowFirstLLM:
        """Delays the first task so it would finish last if order were racy."""

        def __init__(self):
            self.lock = threading.Lock()

        def generate(self, prompt, schema):
            if "Task 0" in prompt:
                time.sleep(0.05)
            with self.lock:
                pass
            index = prompt.split("Title: Task ")[1].split("\n")[0]
            return make_analysis(f"t{index}")

    tasks = [make_task(f"t{i}", f"Task {i}") for i in range(4)]
    update = nodes.analyze_tasks(
        AgentState(raw_tasks=tasks), llm_client=SlowFirstLLM(), max_workers=4
    )

    assert [a.task_id for a in update["analyses"]] == ["t0", "t1", "t2", "t3"]


def test_parallel_analysis_runs_concurrently():
    """Four 50ms calls with four workers should take well under 200ms."""

    class SleepyLLM:
        def generate(self, prompt, schema):
            time.sleep(0.05)
            return make_analysis()

    tasks = [make_task(f"t{i}", f"Task {i}") for i in range(4)]
    started = time.perf_counter()
    nodes.analyze_tasks(AgentState(raw_tasks=tasks), llm_client=SleepyLLM(), max_workers=4)
    assert time.perf_counter() - started < 0.15


# --- plan_week -----------------------------------------------------------


def _plan(*task_ids):
    return WeeklyPlan(
        days=[
            DayPlan(
                day="Monday",
                focus="f",
                tasks=[PlannedTask(task_id=tid, action_summary="do") for tid in task_ids],
            )
        ],
        notes="n",
    )


def test_plan_week_drops_task_ids_that_do_not_exist():
    state = AgentState(
        raw_tasks=[make_task("t1")],
        analyses=[make_analysis("t1")],
        recommendation=Recommendation(ranked_tasks=["t1"], reasoning="r"),
    )
    llm = FakeLLM(_plan("t1", "invented"))

    update = nodes.plan_week(state, llm_client=llm, days=["Monday"], capacity_minutes=180)

    assert [t.task_id for t in update["plan"].days[0].tasks] == ["t1"]


def test_plan_week_plans_in_recommended_order():
    """The prompt lists tasks in ranked order, not fetch order."""
    state = AgentState(
        raw_tasks=[make_task("t1", "First"), make_task("t2", "Second")],
        analyses=[make_analysis("t1"), make_analysis("t2")],
        recommendation=Recommendation(ranked_tasks=["t2", "t1"], reasoning="r"),
    )
    llm = FakeLLM(_plan("t2"))

    nodes.plan_week(state, llm_client=llm, days=["Monday"], capacity_minutes=180)

    prompt = llm.calls[0]
    assert prompt.index("id=t2") < prompt.index("id=t1")
    assert "Monday" in prompt and "180" in prompt


def test_plan_week_skips_the_model_with_nothing_to_plan():
    llm = FakeLLM()
    update = nodes.plan_week(AgentState(), llm_client=llm, days=["Monday"], capacity_minutes=180)
    assert update["plan"] is None
    assert llm.calls == []


def test_plan_week_records_model_failures():
    state = AgentState(raw_tasks=[make_task("t1")], analyses=[make_analysis("t1")])
    llm = FakeLLM(LLMError("quota"), LLMError("quota"))

    update = nodes.plan_week(state, llm_client=llm, days=["Monday"], capacity_minutes=180)

    assert update["plan"] is None
    assert "quota" in update["errors"][0]
