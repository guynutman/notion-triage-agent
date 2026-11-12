"""The shared-types core: every data shape and every boundary contract.

Four families live here: what the Notion API gives us (NotionTask), what
the model is asked to produce (Classification through WeeklyPlan), what
flows between pipeline steps (AgentState), and the protocols and errors of
the two outside services (TaskSource, LLMClient, NotionAPIError, LLMError).

Every other module imports from here and from nowhere sideways: nodes.py
never imports llm.py or notion_client.py, and those two never import each
other. The adapters implement the protocols; the nodes consume them.

Descriptions on the LLM-output models are emitted into the JSON schema the
model receives, so they are prompt text, not documentation. No I/O, no
imports from other project modules.
"""

import operator
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Protocol, TypeVar

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema


class TaskCategory(StrEnum):
    """What kind of item this is."""

    TASK = "task"
    REFERENCE = "reference"
    IDEA = "idea"


class PriorityLevel(StrEnum):
    """Urgency ranking assigned by the model."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AnalysisStatus(StrEnum):
    """Whether a TaskAnalysis came from the model or from the fallback path."""

    ANALYZED = "analyzed"
    NEEDS_REVIEW = "needs_review"


class NotionTask(BaseModel):
    """A row fetched from Notion, before any AI processing."""

    id: str
    title: str
    description: str | None = None
    status: str | None = None
    created_at: datetime | None = None
    url: str | None = None
    # Escape hatch: the full property dict, so columns we do not model yet
    # still reach the caller.
    raw_properties: dict[str, Any] = Field(default_factory=dict)


class Classification(BaseModel):
    """LLM output: which category an item belongs to."""

    category: TaskCategory = Field(
        description="'task' if the item requires the user to do concrete work, "
        "'reference' if it is information to file or read but not act on, "
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
    """A single concrete action extracted from a task."""

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
    """LLM output: the full analysis of one task."""

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
    # Set by the pipeline, never by the model: SkipJsonSchema keeps these out
    # of the schema the model is shown, so it cannot claim a status.
    status: SkipJsonSchema[AnalysisStatus] = AnalysisStatus.ANALYZED
    error: SkipJsonSchema[str | None] = None

    @property
    def needs_review(self) -> bool:
        return self.status is AnalysisStatus.NEEDS_REVIEW

    @classmethod
    def fallback(cls, task_id: str, error: str) -> "TaskAnalysis":
        """A placeholder for a task the model could not analyze.

        Keeps the task in the ranking and the report, flagged for manual
        review, instead of silently dropping it. Priority is the neutral
        'medium' and confidence is 0.0, so nothing downstream mistakes it
        for a real judgement.
        """
        return cls(
            task_id=task_id,
            classification=Classification(
                category=TaskCategory.TASK,
                confidence=0.0,
                reasoning="Not analyzed: the model failed to produce a valid result.",
            ),
            action_items=[],
            priority=PriorityLevel.MEDIUM,
            priority_reasoning="Automatic analysis failed; review this task manually.",
            status=AnalysisStatus.NEEDS_REVIEW,
            error=error,
        )


class Recommendation(BaseModel):
    """LLM output: what to work on next, and in what order."""

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
        description="What to actually do in this sitting, in one short imperative line. "
        "Name the specific step for this day (e.g. 'Draft sections 1-2'), "
        "not the task title -- the title is already shown next to it."
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
    """The state passed through the LangGraph pipeline.

    Each node reads what it needs and returns only the fields it changed.
    """

    raw_tasks: list[NotionTask] = Field(default_factory=list)
    analyses: list[TaskAnalysis] = Field(default_factory=list)
    recommendation: Recommendation | None = None
    plan: WeeklyPlan | None = None
    # operator.add is a LangGraph reducer: node updates are appended to this
    # list instead of replacing it, so errors from every node survive the run.
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)


# --- service boundaries ---------------------------------------------------

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Raised when the model call fails or returns unusable output.

    `retry_after` carries the server's requested wait in seconds when it sent
    one, so the caller can back off for exactly as long as it was told to
    rather than guessing.
    """

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class NotionAPIError(Exception):
    """Raised when the Notion API returns a non-2xx response."""


class LLMClient(Protocol):
    """What the nodes need from a language model. Nothing else."""

    def generate(self, prompt: str, schema: type[T]) -> T:
        """Return an instance of `schema` produced from `prompt`.

        Raises LLMError on any failure, including output that does not
        validate against `schema`.
        """
        ...


class TaskSource(Protocol):
    """What the nodes need from the task database. Nothing else."""

    def fetch_tasks(
        self,
        filter_status: str | None = None,
        limit: int | None = None,
        exclude_done: bool = True,
    ) -> list[NotionTask]: ...

    def ensure_select_properties(self, names: list[str]) -> list[str]: ...

    def update_task_properties(self, task_id: str, properties: dict) -> None: ...
