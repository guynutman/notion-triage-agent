"""Notion API wrapper.

Knows about HTTP and Notion's JSON shapes. Returns NotionTask models.
Never calls an LLM, never reads environment variables.
"""

from datetime import datetime

import httpx

from notion_triage_agent.models import NotionTask

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"

# v1 assumes these column names. TODO: make configurable.
TITLE_PROPERTY = "Name"
DESCRIPTION_PROPERTY = "Description"
STATUS_PROPERTY = "Status"


class NotionAPIError(Exception):
    """Raised when the Notion API returns a non-2xx response."""


class NotionClient:
    """Wrapper around the Notion API. Handles auth, pagination, property parsing."""

    def __init__(self, token: str, database_id: str) -> None:
        """Store credentials and build auth headers. No network calls here."""
        self._database_id = database_id
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }

    def fetch_tasks(self, filter_status: str | None = None) -> list[NotionTask]:
        """Query the database and return every matching row as a NotionTask.

        POSTs to /databases/{id}/query. Follows pagination until has_more is
        false, passing next_cursor as start_cursor. Returns a flat list --
        the caller never sees a cursor.

        If filter_status is given, adds:
            {"filter": {"property": STATUS_PROPERTY, "select": {"equals": filter_status}}}

        Raises NotionAPIError on any non-2xx response.
        """
        url = f"{NOTION_API_BASE}/databases/{self._database_id}/query"
        body: dict = {}
        if filter_status is not None:
            body["filter"] = {
                "property": STATUS_PROPERTY,
                "select": {"equals": filter_status},
            }

        tasks: list[NotionTask] = []
        while True:
            response = httpx.post(url, headers=self._headers, json=body)
            data = self._json_or_raise(response)

            tasks.extend(self._parse_page(page) for page in data["results"])

            if not data.get("has_more"):
                return tasks
            body["start_cursor"] = data["next_cursor"]

    def update_task_properties(self, task_id: str, properties: dict) -> None:
        """PATCH /pages/{task_id} with {"properties": properties}.

        Used later to write category/priority back to Notion.
        Raises NotionAPIError on failure.
        """
        response = httpx.patch(
            f"{NOTION_API_BASE}/pages/{task_id}",
            headers=self._headers,
            json={"properties": properties},
        )
        self._json_or_raise(response)

    @staticmethod
    def _json_or_raise(response: httpx.Response) -> dict:
        """Return the parsed body, or raise NotionAPIError on a non-2xx response."""
        if response.is_success:
            return response.json()
        try:
            message = response.json().get("message", response.text)
        except ValueError:  # error body was not JSON
            message = response.text
        raise NotionAPIError(f"Notion API {response.status_code}: {message}")

    def _parse_page(self, page: dict) -> NotionTask:
        """Convert one raw Notion page dict into a NotionTask.

        Pure function -- no network access, so it is testable against a fixture.

        Page-level fields live outside "properties": id, url, created_time.
        Everything not extracted into a typed field goes into raw_properties.

        Edge cases that must not raise:
          - property key missing entirely
          - "select" is null (nobody picked a status)
          - "rich_text"/"title" is an empty list
          - title split across multiple segments -> join every plain_text
        """
        properties = page.get("properties", {})

        created_time = page.get("created_time")
        created_at = datetime.fromisoformat(created_time) if created_time else None

        title = self._plain_text(properties.get(TITLE_PROPERTY), "title")

        return NotionTask(
            id=page["id"],
            title=title or "(untitled)",
            description=self._plain_text(properties.get(DESCRIPTION_PROPERTY), "rich_text"),
            status=self._select_name(properties.get(STATUS_PROPERTY)),
            created_at=created_at,
            url=page.get("url"),
            raw_properties=properties,
        )

    @staticmethod
    def _plain_text(prop: dict | None, key: str) -> str | None:
        """Join plain_text across all segments of a title or rich_text property.

        key is "title" or "rich_text". Returns None when the property is
        missing or empty -- never an empty string.
        """
        if not prop:
            return None
        segments = prop.get(key) or []
        text = "".join(segment["plain_text"] for segment in segments)
        return text or None

    @staticmethod
    def _select_name(prop: dict | None) -> str | None:
        """Return the name of a select property, or None if unset or missing."""
        if not prop:
            return None
        select = prop.get("select")
        if select is None:
            return None
        return select.get("name")
