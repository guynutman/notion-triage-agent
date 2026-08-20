"""End-to-end pipeline test with fake clients.

Runs the compiled LangGraph -- fetch, analyze, recommend -- with no network
and no API key, which is only possible because build_graph takes its
dependencies as arguments.
"""

from notion_triage_agent.graph import build_graph
from notion_triage_agent.llm import LLMError
from notion_triage_agent.models import AgentState, Recommendation, TaskAnalysis
from notion_triage_agent.notion_client import NotionAPIError
from tests.test_nodes import FakeLLM, FakeNotion, make_analysis, make_task


class SchemaAwareLLM:
    """Returns a canned value chosen by the requested schema."""

    def __init__(self, analysis: TaskAnalysis, recommendation: Recommendation):
        self._by_schema = {
            TaskAnalysis: analysis,
            Recommendation: recommendation,
        }

    def generate(self, prompt: str, schema):
        return self._by_schema[schema]


def test_pipeline_runs_start_to_end():
    graph = build_graph(
        FakeNotion([make_task("t1")]),
        SchemaAwareLLM(make_analysis("t1"), Recommendation(ranked_tasks=["t1"], reasoning="r")),
    )

    state = AgentState.model_validate(graph.invoke(AgentState()))

    assert [task.id for task in state.raw_tasks] == ["t1"]
    assert [analysis.task_id for analysis in state.analyses] == ["t1"]
    assert state.recommendation.ranked_tasks == ["t1"]
    assert state.errors == []


def test_errors_accumulate_across_nodes():
    """The Annotated[..., operator.add] reducer on `errors` is what makes a
    later node's update append rather than overwrite an earlier one."""
    graph = build_graph(
        FakeNotion(error=NotionAPIError("401 bad token")),
        FakeLLM(LLMError("never reached")),
    )

    state = AgentState.model_validate(graph.invoke(AgentState()))

    # fetch_tasks failed; analyze_tasks then ran on zero tasks without
    # clearing the error it left behind.
    assert len(state.errors) == 1
    assert "401 bad token" in state.errors[0]
    assert state.recommendation is None


def test_empty_fetch_skips_the_model_entirely():
    """The conditional edge after fetch_tasks routes an empty database to END."""
    llm = FakeLLM()  # any call would raise "fake ran out of responses"
    graph = build_graph(FakeNotion([]), llm)

    state = AgentState.model_validate(graph.invoke(AgentState()))

    assert state.analyses == []
    assert state.recommendation is None
    assert state.errors == []
    assert llm.calls == []


def test_write_back_runs_only_when_enabled():
    notion = FakeNotion([make_task("t1")])
    llm = SchemaAwareLLM(make_analysis("t1"), Recommendation(ranked_tasks=["t1"], reasoning="r"))

    build_graph(notion, llm).invoke(AgentState())
    assert notion.updates == []

    build_graph(notion, llm, write_back=True).invoke(AgentState())
    assert [task_id for task_id, _ in notion.updates] == ["t1"]


def test_run_options_reach_the_fetch_node():
    notion = FakeNotion([make_task("t1")])
    llm = SchemaAwareLLM(make_analysis("t1"), Recommendation(ranked_tasks=["t1"], reasoning="r"))

    build_graph(notion, llm, filter_status="Not started", limit=5).invoke(AgentState())

    assert notion.fetch_kwargs == {
        "filter_status": "Not started",
        "limit": 5,
        "exclude_done": True,
    }
