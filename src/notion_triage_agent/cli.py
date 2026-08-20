"""Command-line entry point.

The only module that reads environment variables, prints to stdout, or has
a __main__ block. Builds the dependencies, runs the graph, formats output.
"""

import argparse
import os
import sys
from datetime import date

from dotenv import load_dotenv

from notion_triage_agent.graph import build_graph
from notion_triage_agent.llm import (
    DEFAULT_MODEL,
    DEFAULT_REQUESTS_PER_MINUTE,
    GeminiClient,
)
from notion_triage_agent.models import AgentState, TaskAnalysis
from notion_triage_agent.nodes import (
    DEFAULT_CAPACITY_MINUTES,
    DEFAULT_WORKERS,
    WEEKDAYS,
)
from notion_triage_agent.notion_client import NotionClient

REQUIRED_VARS = ("NOTION_TOKEN", "NOTION_DATABASE_ID", "GEMINI_API_KEY")

PRIORITY_ICONS = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "⚪",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options.

    Takes argv so the parser is testable without touching sys.argv.
    """
    parser = argparse.ArgumentParser(
        prog="notion-triage-agent",
        description="Triage a Notion task database and rank what to work on next.",
    )
    parser.add_argument(
        "--status",
        metavar="NAME",
        help='only triage rows with this Status, e.g. "Not started"',
    )
    parser.add_argument("--limit", type=int, metavar="N", help="triage at most N tasks")
    parser.add_argument(
        "--include-done",
        action="store_true",
        help='also triage rows whose Status is "Done" (excluded by default)',
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=f"parallel model calls during analysis (default {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--write-back",
        action="store_true",
        help="write the category and priority back into Notion",
    )
    parser.add_argument(
        "--plan",
        nargs="?",
        const=len(WEEKDAYS),
        type=int,
        metavar="DAYS",
        help=f"also schedule the tasks across the next DAYS days "
        f"(default {len(WEEKDAYS)}, starting today)",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=DEFAULT_CAPACITY_MINUTES,
        metavar="MINUTES",
        help=f"focused minutes available per day when planning "
        f"(default {DEFAULT_CAPACITY_MINUTES})",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
        metavar="NAME",
        help=f"Gemini model to use (default {DEFAULT_MODEL}, or $GEMINI_MODEL). "
        f"Free-tier daily quotas differ sharply between models.",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=DEFAULT_REQUESTS_PER_MINUTE,
        metavar="N",
        help=f"model requests per minute to stay under "
        f"(default {DEFAULT_REQUESTS_PER_MINUTE}, the Gemini free-tier limit)",
    )
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.plan is not None and not 1 <= args.plan <= 14:
        parser.error("--plan must be between 1 and 14 days")
    if args.capacity < 1:
        parser.error("--capacity must be at least 1 minute")
    if args.rpm < 1:
        parser.error("--rpm must be at least 1")
    return args


def upcoming_days(count: int, today: date | None = None) -> list[str]:
    """Day names for the next `count` days, starting today.

    Takes `today` so the result is testable without freezing the clock.
    """
    start = today or date.today()
    return [WEEKDAYS[(start.weekday() + offset) % 7] for offset in range(count)]


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


def format_results(state: AgentState, capacity_minutes: int | None = None) -> str:
    """Render the finished state as the CLI report.

    capacity_minutes is only used to flag days the planner overfilled.
    """
    if not state.analyses:
        lines = ["No tasks to triage."]
        return "\n".join(lines + _format_errors(state))

    by_id = {analysis.task_id: analysis for analysis in state.analyses}
    titles = {task.id: task.title for task in state.raw_tasks}
    order = state.recommendation.ranked_tasks if state.recommendation else list(by_id)

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
            lines.append(
                f"   Total estimated: ~{total} min of work across {len(state.analyses)} tasks"
            )

    if state.plan:
        lines += _format_plan(state.plan, titles, capacity_minutes)

    return "\n".join(lines + _format_errors(state))


def _format_plan(plan, titles: dict, capacity_minutes: int | None) -> list[str]:
    """Render the weekly schedule.

    Scheduled minutes can exceed the recommendation's total: a task too big
    for one sitting is split across days and counted once per session. The
    footer says so, because two different totals with no explanation reads
    as a bug.
    """
    lines = ["", "🗓  Plan"]
    scheduled = 0
    sessions = 0
    for day in plan.days:
        total = sum(task.estimated_minutes or 0 for task in day.tasks)
        scheduled += total
        sessions += len(day.tasks)
        over = capacity_minutes is not None and total > capacity_minutes
        header = f"   {day.day}" + (f"  (~{total} min)" if total else "")
        if over:
            header += f"  ⚠ over the {capacity_minutes} min capacity"
        lines.append(header)
        lines.append(f"      {day.focus}")
        for task in day.tasks:
            estimate = f" (~{task.estimated_minutes} min)" if task.estimated_minutes else ""
            title = titles.get(task.task_id, "(unknown)")
            lines.append(f"      • {title}: {task.action_summary}{estimate}")
        if not day.tasks:
            lines.append("      • (open)")
        lines.append("")
    if scheduled:
        lines.append(
            f"   Scheduled: ~{scheduled} min over {sessions} "
            f"session{'s' if sessions != 1 else ''} "
            f"(large tasks are split across days)."
        )
    lines.append(f"   {plan.notes}")
    return lines


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
            estimate = f" (~{item.estimated_minutes} min)" if item.estimated_minutes else ""
            lines.append(f"      • {item.description}{estimate}")
    lines.append("")
    return lines


def _format_errors(state: AgentState) -> list[str]:
    """Render any errors collected during the run."""
    if not state.errors:
        return []
    return ["", f"⚠️  {len(state.errors)} error(s):"] + [f"   - {error}" for error in state.errors]


def main() -> None:
    """Parse args, build dependencies, run the graph, print the report."""
    args = parse_args()
    try:
        config = load_config()
    except RuntimeError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    notion_client = NotionClient(config["NOTION_TOKEN"], config["NOTION_DATABASE_ID"])
    llm_client = GeminiClient(
        config["GEMINI_API_KEY"], model=args.model, requests_per_minute=args.rpm
    )

    graph = build_graph(
        notion_client,
        llm_client,
        filter_status=args.status,
        limit=args.limit,
        exclude_done=not args.include_done,
        max_workers=args.workers,
        write_back=args.write_back,
        plan_days=upcoming_days(args.plan) if args.plan else None,
        capacity_minutes=args.capacity,
    )
    final_state = AgentState.model_validate(graph.invoke(AgentState()))

    print(format_results(final_state, capacity_minutes=args.capacity))


if __name__ == "__main__":
    main()
