"""LangGraph wiring.

Pure orchestration: no LLM calls, no HTTP, no logic. Read this file to see
the pipeline at a glance; read nodes.py to see what each step does.
"""

from functools import partial

from langgraph.graph import END, START, StateGraph

from notion_triage_agent import nodes
from notion_triage_agent.llm import LLMClient
from notion_triage_agent.models import AgentState
from notion_triage_agent.notion_client import NotionClient


def build_graph(notion_client: NotionClient, llm_client: LLMClient):
    """Wire the triage pipeline and compile it.

        START -> fetch_tasks -> analyze_tasks -> recommend -> END

    Dependencies are bound to the nodes with functools.partial, so LangGraph
    only ever passes state. Returns a compiled graph: call
    .invoke(AgentState()) to run it.
    """
    builder = StateGraph(AgentState)

    builder.add_node("fetch_tasks", partial(nodes.fetch_tasks, notion_client=notion_client))
    builder.add_node("analyze_tasks", partial(nodes.analyze_tasks, llm_client=llm_client))
    builder.add_node("recommend", partial(nodes.recommend, llm_client=llm_client))

    # Linear in v1. A conditional edge would go after fetch_tasks -- skip
    # straight to END when no tasks came back, rather than calling the model.
    builder.add_edge(START, "fetch_tasks")
    builder.add_edge("fetch_tasks", "analyze_tasks")
    builder.add_edge("analyze_tasks", "recommend")
    builder.add_edge("recommend", END)

    return builder.compile()
