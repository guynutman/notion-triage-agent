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


def build_graph(
    notion_client: NotionClient,
    llm_client: LLMClient,
    *,
    filter_status: str | None = None,
    limit: int | None = None,
    max_workers: int = nodes.DEFAULT_WORKERS,
    write_back: bool = False,
    plan_days: list[str] | None = None,
    capacity_minutes: int = nodes.DEFAULT_CAPACITY_MINUTES,
):
    """Wire the triage pipeline and compile it.

        START -> fetch_tasks -+-> analyze_tasks -> recommend -> [plan_week]
                              |                                      |
                              +-> END (nothing fetched)        [write_back] -> END

    Run options are bound here rather than carried in the state: they are
    fixed for the whole run, and keeping them out of AgentState means no node
    can quietly depend on configuration it was not handed.

    Dependencies are bound to the nodes with functools.partial, so LangGraph
    only ever passes state. Returns a compiled graph: call
    .invoke(AgentState()) to run it.
    """
    builder = StateGraph(AgentState)

    builder.add_node(
        "fetch_tasks",
        partial(
            nodes.fetch_tasks,
            notion_client=notion_client,
            filter_status=filter_status,
            limit=limit,
        ),
    )
    builder.add_node(
        "analyze_tasks",
        partial(nodes.analyze_tasks, llm_client=llm_client, max_workers=max_workers),
    )
    builder.add_node("recommend", partial(nodes.recommend, llm_client=llm_client))

    builder.add_edge(START, "fetch_tasks")

    # An empty database costs nothing: skip straight to END rather than
    # sending an empty prompt to the model.
    builder.add_conditional_edges(
        "fetch_tasks",
        nodes.has_tasks,
        {"analyze_tasks": "analyze_tasks", "__end__": END},
    )
    builder.add_edge("analyze_tasks", "recommend")

    # Optional stages are appended in order, so the chain stays linear
    # whichever combination of flags is used.
    tail = "recommend"
    if plan_days:
        builder.add_node(
            "plan_week",
            partial(
                nodes.plan_week,
                llm_client=llm_client,
                days=plan_days,
                capacity_minutes=capacity_minutes,
            ),
        )
        builder.add_edge(tail, "plan_week")
        tail = "plan_week"
    if write_back:
        builder.add_node("write_back", partial(nodes.write_back, notion_client=notion_client))
        builder.add_edge(tail, "write_back")
        tail = "write_back"
    builder.add_edge(tail, END)

    return builder.compile()
