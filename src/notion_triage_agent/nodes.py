"""LangGraph nodes -- one step of the pipeline each.

Every node takes the current AgentState and returns a dict of only the
fields it changed; LangGraph merges that into the state.

Dependencies (notion_client, llm_client) arrive as keyword arguments so the
nodes can be tested with fakes. graph.py binds the real ones with
functools.partial.
"""

import time
from concurrent.futures import ThreadPoolExecutor

from notion_triage_agent.llm import LLMClient, LLMError
from notion_triage_agent.models import (
    AgentState,
    NotionTask,
    Recommendation,
    TaskAnalysis,
    WeeklyPlan,
)
from notion_triage_agent.notion_client import NotionAPIError, NotionClient

DEFAULT_WORKERS = 4
BACKOFF_SECONDS = 2.0
RETRY_ATTEMPTS = 3
DEFAULT_CAPACITY_MINUTES = 180
WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

# Properties written back to Notion when --write-back is used.
CATEGORY_PROPERTY = "AI Category"
PRIORITY_PROPERTY = "AI Priority"

ANALYZE_PROMPT = """You are triaging one item from a personal task database.

Title: {title}
Description: {description}
Status: {status}

Classify the item, extract its concrete action items, and assign a priority.
Set task_id to exactly: {task_id}
Base your reasoning only on the text above. Do not invent details."""

RECOMMEND_PROMPT = """You are advising on what to work on next.

Here are the analyzed tasks:

{task_summaries}

Rank them in the order they should be worked on, best first. Consider
priority, whether an item blocks other work, and how well scoped it is.
Use only the task IDs listed above."""

PLAN_PROMPT = """You are laying out a work week.

Days available: {days}
Daily capacity: about {capacity} minutes of focused work.

Tasks, already ranked best-first:

{task_summaries}

Distribute the tasks across the days. Front-load the high-priority and
blocking work. Respect the daily capacity: it is better to leave a day
lighter than to overload it, and a task with no estimate should be assumed
to take about 45 minutes. A task may appear on more than one day if it is
too large for a single sitting. Do not invent tasks or IDs."""


def fetch_tasks(
    state: AgentState,
    *,
    notion_client: NotionClient,
    filter_status: str | None = None,
    limit: int | None = None,
) -> dict:
    """Node 1: load tasks from Notion.

    Returns {"raw_tasks": [...]}. On API failure, records the error and
    returns no tasks so the pipeline can finish cleanly.
    """
    try:
        tasks = notion_client.fetch_tasks(filter_status=filter_status, limit=limit)
    except NotionAPIError as exc:
        return {"raw_tasks": [], "errors": [f"fetch_tasks: {exc}"]}
    return {"raw_tasks": tasks}


def analyze_tasks(
    state: AgentState, *, llm_client: LLMClient, max_workers: int = DEFAULT_WORKERS
) -> dict:
    """Node 2: analyze every fetched task, one LLM call each.

    One task failing does not abort the batch -- its error is recorded and
    the remaining tasks are still analyzed.

    Tasks are independent, so the calls run on a thread pool: the work is
    network-bound, so threads are enough and asyncio is not needed. Results
    are collected in submission order, which keeps output deterministic no
    matter what order the responses arrive in. Pass max_workers=1 to force
    sequential execution.
    """
    if not state.raw_tasks:
        return {"analyses": [], "errors": []}

    analyses: list[TaskAnalysis] = []
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=min(max_workers, len(state.raw_tasks))) as pool:
        futures = [pool.submit(_analyze_one, llm_client, task) for task in state.raw_tasks]
        for task, future in zip(state.raw_tasks, futures, strict=True):
            try:
                analyses.append(future.result())
            except LLMError as exc:
                errors.append(f"analyze_tasks[{task.title}]: {exc}")

    return {"analyses": analyses, "errors": errors}


def _analyze_one(llm_client: LLMClient, task: NotionTask) -> TaskAnalysis:
    """Analyze a single task. Runs on a worker thread."""
    prompt = ANALYZE_PROMPT.format(
        title=task.title,
        description=task.description or "(none)",
        status=task.status or "(unset)",
        task_id=task.id,
    )
    analysis = _generate_with_retry(llm_client, prompt, TaskAnalysis)
    # The model is told to echo the task id, but is not trusted to.
    analysis.task_id = task.id
    return analysis


def recommend(state: AgentState, *, llm_client: LLMClient) -> dict:
    """Node 3: rank the analyzed tasks into a recommendation.

    Returns {"recommendation": None} when there is nothing to rank.
    Hallucinated task IDs are dropped, and any analyzed task the model
    omitted is appended, so ranked_tasks always matches the real analyses.
    """
    if not state.analyses:
        return {"recommendation": None}

    prompt = RECOMMEND_PROMPT.format(
        task_summaries="\n".join(
            _summarize(analysis, state.raw_tasks) for analysis in state.analyses
        )
    )
    try:
        recommendation = _generate_with_retry(llm_client, prompt, Recommendation)
    except LLMError as exc:
        return {"recommendation": None, "errors": [f"recommend: {exc}"]}

    known_ids = [analysis.task_id for analysis in state.analyses]
    ranked = [task_id for task_id in recommendation.ranked_tasks if task_id in known_ids]
    ranked += [task_id for task_id in known_ids if task_id not in ranked]
    recommendation.ranked_tasks = ranked

    return {"recommendation": recommendation}


def write_back(state: AgentState, *, notion_client: NotionClient) -> dict:
    """Optional node 4: write each task's category and priority into Notion.

    The select columns are created if the database does not have them, so a
    first run against a plain task database works without manual setup.

    One failed page does not stop the rest; every failure is recorded.
    """
    if not state.analyses:
        return {}

    errors: list[str] = []
    try:
        notion_client.ensure_select_properties([CATEGORY_PROPERTY, PRIORITY_PROPERTY])
    except NotionAPIError as exc:
        return {"errors": [f"write_back: could not prepare properties: {exc}"]}

    for analysis in state.analyses:
        try:
            notion_client.update_task_properties(
                analysis.task_id,
                {
                    CATEGORY_PROPERTY: {"select": {"name": analysis.classification.category.value}},
                    PRIORITY_PROPERTY: {"select": {"name": analysis.priority.value}},
                },
            )
        except NotionAPIError as exc:
            errors.append(f"write_back[{analysis.task_id}]: {exc}")

    return {"errors": errors}


def plan_week(
    state: AgentState,
    *,
    llm_client: LLMClient,
    days: list[str],
    capacity_minutes: int,
) -> dict:
    """Optional node: schedule the ranked tasks across the coming days.

    Runs after `recommend` so the model plans against an order that has
    already been reconciled against the real task list. Task IDs are checked
    again here -- a plan that references a task you do not have is worse than
    no plan.
    """
    if not state.analyses:
        return {"plan": None}

    ordered = _ranked_analyses(state)
    prompt = PLAN_PROMPT.format(
        days=", ".join(days),
        capacity=capacity_minutes,
        task_summaries="\n".join(_summarize(analysis, state.raw_tasks) for analysis in ordered),
    )
    try:
        plan = _generate_with_retry(llm_client, prompt, WeeklyPlan)
    except LLMError as exc:
        return {"plan": None, "errors": [f"plan_week: {exc}"]}

    known_ids = {analysis.task_id for analysis in state.analyses}
    for day in plan.days:
        day.tasks = [task for task in day.tasks if task.task_id in known_ids]

    return {"plan": plan}


def _ranked_analyses(state: AgentState) -> list[TaskAnalysis]:
    """Analyses in recommended order, falling back to fetch order."""
    if not state.recommendation:
        return state.analyses
    by_id = {analysis.task_id: analysis for analysis in state.analyses}
    return [by_id[task_id] for task_id in state.recommendation.ranked_tasks if task_id in by_id]


def has_tasks(state: AgentState) -> str:
    """Conditional-edge router: skip the model entirely on an empty fetch."""
    return "analyze_tasks" if state.raw_tasks else "__end__"


def _generate_with_retry(
    llm_client: LLMClient, prompt: str, schema, attempts: int = RETRY_ATTEMPTS
):
    """Call the model, retrying with backoff before giving up.

    Structured-output failures are usually transient -- a truncated response,
    a malformed field, a rate limit -- so retrying is cheap and often works.
    Retrying *immediately* is not: a quota error re-fires into the same closed
    window. So we wait for exactly as long as the server asked when it told
    us (`retry_after`), and fall back to exponential backoff when it did not.
    """
    last_error: LLMError | None = None
    for attempt in range(attempts):
        try:
            return llm_client.generate(prompt, schema)
        except LLMError as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            delay = getattr(exc, "retry_after", None)
            time.sleep(delay if delay is not None else BACKOFF_SECONDS * 2**attempt)
    raise last_error


def _summarize(analysis: TaskAnalysis, raw_tasks: list[NotionTask]) -> str:
    """One line per task for the recommendation prompt."""
    titles = {task.id: task.title for task in raw_tasks}
    minutes = sum(item.estimated_minutes or 0 for item in analysis.action_items)
    return (
        f"- id={analysis.task_id} | {titles.get(analysis.task_id, '(unknown)')} "
        f"| priority={analysis.priority.value} "
        f"| category={analysis.classification.category.value} "
        f"| {len(analysis.action_items)} actions (~{minutes} min)"
    )
