"""LangGraph nodes -- one step of the pipeline each.

Every node takes the current AgentState and returns a dict of only the
fields it changed; LangGraph merges that into the state.

Dependencies (notion_client, llm_client) arrive as keyword arguments so the
nodes can be tested with fakes. graph.py binds the real ones with
functools.partial.
"""

from notion_triage_agent.llm import LLMClient, LLMError
from notion_triage_agent.models import (
    AgentState,
    NotionTask,
    Recommendation,
    TaskAnalysis,
)
from notion_triage_agent.notion_client import NotionAPIError, NotionClient

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


def fetch_tasks(state: AgentState, *, notion_client: NotionClient) -> dict:
    """Node 1: load tasks from Notion.

    Returns {"raw_tasks": [...]}. On API failure, records the error and
    returns no tasks so the pipeline can finish cleanly.
    """
    try:
        tasks = notion_client.fetch_tasks()
    except NotionAPIError as exc:
        return {"raw_tasks": [], "errors": [f"fetch_tasks: {exc}"]}
    return {"raw_tasks": tasks}


def analyze_tasks(state: AgentState, *, llm_client: LLMClient) -> dict:
    """Node 2: analyze every fetched task, one LLM call each.

    One task failing does not abort the batch -- its error is recorded and
    the remaining tasks are still analyzed.

    Sequential in v1. Tasks are independent, so this is the natural place to
    parallelize (thread pool or asyncio.gather) once latency matters.
    """
    analyses: list[TaskAnalysis] = []
    errors: list[str] = []

    for task in state.raw_tasks:
        prompt = ANALYZE_PROMPT.format(
            title=task.title,
            description=task.description or "(none)",
            status=task.status or "(unset)",
            task_id=task.id,
        )
        try:
            analysis = _generate_with_retry(llm_client, prompt, TaskAnalysis)
        except LLMError as exc:
            errors.append(f"analyze_tasks[{task.title}]: {exc}")
            continue

        # The model is told to echo the task id, but is not trusted to.
        analysis.task_id = task.id
        analyses.append(analysis)

    return {"analyses": analyses, "errors": errors}


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


def _generate_with_retry(llm_client: LLMClient, prompt: str, schema, attempts: int = 2):
    """Call the model, retrying once before giving up.

    Structured-output failures are usually transient -- a truncated response
    or a malformed field -- so a second attempt is cheap and often works.
    """
    last_error: LLMError | None = None
    for _ in range(attempts):
        try:
            return llm_client.generate(prompt, schema)
        except LLMError as exc:
            last_error = exc
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
