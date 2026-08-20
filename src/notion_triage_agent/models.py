import operator
from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field


class TaskCategory(StrEnum):
    TASK = "task"
    REFERENCE = "reference"
    IDEA = "idea"


class PriorityLevel(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class NotionTask(BaseModel):
    id: str
    title: str
    description: str | None = None
    status: str | None = None
    raw_properties: dict = Field(default_factory=dict)
    created_at: datetime | None = None
    url: str | None = None


class Classification(BaseModel):
    category: TaskCategory = Field(
        description="'task' if the item requires the user to do concrete work, "
        "'reference' if it is information to keep but act on, "
        "'idea' if it is a possibility to explore later with no committed work."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How certain you are of the category, from 0.0 to 1.0. "
        "Use below 0.6 when the item is ambiguous or too short to judge.",
    )
    reasoning: str = Field(
        description="One sentence explaining the category choice, "
        "citing specific wording from the task."
    )


class ActionItem(BaseModel):
    description: str = Field(
        description="A single concrete action, starting with a verb. "
        "Split multi-step work into separate items rather than combining them."
    )
    estimated_minutes: int | None = Field(
        default=None,
        description="Estimated minutes of focused work for this one action. "
        "Use null if the task lacks enough detail to estimate; do not guess.",
    )


class TaskAnalysis(BaseModel):
    task_id: str = Field(
        description="The Notion page ID of the task being analyzed, copied verbatim."
    )
    classification: Classification
    action_items: list[ActionItem] = Field(
        default_factory=list,
        description="Concrete actions extracted from the task. "
        "Empty list for reference or idea items that require no work yet.",
    )
    priority: PriorityLevel = Field(
        description="'critical' if blocking other work or past due, 'high' if time-sensitive, "
        "'medium' for normal work, 'low' for optional or someday items."
    )
    priority_reasoning: str = Field(
        description="One sentence justifying the priority, "
        "referencing urgency, blockers, or due dates."
    )


class Recommendation(BaseModel):
    ranked_tasks: list[str] = Field(
        default_factory=list,
        description="Task IDs in recommended work order, best first. "
        "Use only IDs present in the supplied analyses; never invent one.",
    )
    reasoning: str = Field(
        description="Two or three sentences explaining the ordering and what to start with."
    )
    estimated_total_minutes: int | None = Field(
        default=None,
        description="Sum of estimated minutes across ranked tasks, "
        "or null if too many estimates are missing.",
    )


class PlannedTask(BaseModel):
    """One task scheduled into a specific day."""

    task_id: str = Field(description="Task ID from the supplied list, copied verbatim.")
    action_summary: str = Field(
        description="What to actually do in this sitting, in one short imperative line."
    )
    estimated_minutes: int | None = Field(
        default=None,
        description="Minutes to reserve for this task on this day, or null if unknown.",
    )


class DayPlan(BaseModel):
    """One day's worth of work."""

    day: str = Field(
        description="Day name, e.g. 'Monday'. Use the days given in the request, in order."
    )
    focus: str = Field(
        description="One sentence naming the point of this day, e.g. what should be "
        "finished by the end of it."
    )
    tasks: list[PlannedTask] = Field(
        default_factory=list,
        description="Tasks to work on this day. Leave empty for a deliberate rest or buffer day.",
    )


class WeeklyPlan(BaseModel):
    """LLM output: the ranked tasks distributed across the week."""

    days: list[DayPlan] = Field(
        default_factory=list,
        description="One entry per day of the week, in order.",
    )
    notes: str = Field(
        description="Two or three sentences on the shape of the week: what was "
        "front-loaded and why, and what to drop first if time runs short."
    )


class AgentState(BaseModel):
    raw_tasks: list[NotionTask] = Field(default_factory=list)
    analyses: list[TaskAnalysis] = Field(default_factory=list)
    recommendation: Recommendation | None = None
    plan: WeeklyPlan | None = None
    # operator.add is a LangGraph reducer: node updates are appended to this
    # list instead of replacing it, so errors from every node survive the run.
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)
