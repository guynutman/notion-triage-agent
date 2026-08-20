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
