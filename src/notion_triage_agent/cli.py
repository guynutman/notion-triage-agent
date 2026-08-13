"""Command-line entry point.

The only module that reads environment variables, prints to stdout, or has
a __main__ block. Builds the dependencies, runs the graph, formats output.
"""

import os
import sys

from dotenv import load_dotenv

from notion_triage_agent.graph import build_graph
from notion_triage_agent.llm import GeminiClient
from notion_triage_agent.models import AgentState, TaskAnalysis
from notion_triage_agent.notion_client import NotionClient

REQUIRED_VARS = ("NOTION_TOKEN", "NOTION_DATABASE_ID", "GEMINI_API_KEY")

PRIORITY_ICONS = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "⚪",
}


def load_config() -> dict:
    """Read credentials from the environment or .env.

    Raises RuntimeError naming every missing variable at once, so you fix
    them in one pass instead of one per run.
    """
    load_dotenv()
    config = {name: os.environ.get(name) for name in REQUIRED_VARS}
    missing = [name for name, value in config.items() if not value]
    if missing:
        raise RuntimeError(
            f"Missing environment variable(s): {', '.join(missing)}. "
            f"Add them to .env in the project root."
        )
    return config


def format_results(state: AgentState) -> str:
    """Render the finished state as the CLI report."""
    if not state.analyses:
        lines = ["No tasks to triage."]
        return "\n".join(lines + _format_errors(state))

    by_id = {analysis.task_id: analysis for analysis in state.analyses}
    titles = {task.id: task.title for task in state.raw_tasks}
    order = (
        state.recommendation.ranked_tasks
        if state.recommendation
        else list(by_id)
    )

    lines = [f"\n📋 Triage Results — {len(state.analyses)} tasks analyzed\n"]
    for position, task_id in enumerate(order, start=1):
        analysis = by_id.get(task_id)
        if analysis is None:
            continue
        lines += _format_task(position, analysis, titles.get(task_id, "(unknown)"))

    if state.recommendation:
        lines.append("💡 Recommendation:")
        lines.append(f"   {state.recommendation.reasoning}")
        total = state.recommendation.estimated_total_minutes
        if total:
            lines.append(f"   Total estimated: ~{total} min")

    return "\n".join(lines + _format_errors(state))


def _format_task(position: int, analysis: TaskAnalysis, title: str) -> list[str]:
    """Render one ranked task as a block of lines."""
    priority = analysis.priority.value
    icon = PRIORITY_ICONS.get(priority, "•")
    classification = analysis.classification

    lines = [
        f"{position:2}. {icon} [{priority.upper()}] {title}",
        f"    Category: {classification.category.value} "
        f"({classification.confidence:.2f} confidence)",
        f"    Why: {analysis.priority_reasoning}",
    ]
    if analysis.action_items:
        lines.append("    Actions:")
        for item in analysis.action_items:
            estimate = (
                f" (~{item.estimated_minutes} min)" if item.estimated_minutes else ""
            )
            lines.append(f"      • {item.description}{estimate}")
    lines.append("")
    return lines


def _format_errors(state: AgentState) -> list[str]:
    """Render any errors collected during the run."""
    if not state.errors:
        return []
    return ["", f"⚠️  {len(state.errors)} error(s):"] + [
        f"   - {error}" for error in state.errors
    ]


def main() -> None:
    """Load config, build dependencies, run the graph, print the report."""
    try:
        config = load_config()
    except RuntimeError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    notion_client = NotionClient(config["NOTION_TOKEN"], config["NOTION_DATABASE_ID"])
    llm_client = GeminiClient(config["GEMINI_API_KEY"])

    graph = build_graph(notion_client, llm_client)
    final_state = AgentState.model_validate(graph.invoke(AgentState()))

    print(format_results(final_state))


if __name__ == "__main__":
    main()
