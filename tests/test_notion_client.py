"""Parsing tests for the Notion client.

_parse_page and its helpers are pure, so every case here runs against a
saved API response with no network access and no credentials.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from notion_triage_agent.notion_client import NotionClient

FIXTURE = Path(__file__).parent / "fixtures" / "notion_query_response.json"


@pytest.fixture
def pages() -> list[dict]:
    return json.loads(FIXTURE.read_text())["results"]


@pytest.fixture
def client() -> NotionClient:
    """A client with fake credentials -- parsing never touches the network."""
    return NotionClient(token="ntn_fake", database_id="fake-db-id")


def test_parses_a_real_page(client, pages):
    task = client._parse_page(pages[0])
    assert task.id == pages[0]["id"]
    assert task.title == "Meditate"
    assert isinstance(task.created_at, datetime)
    assert task.url.startswith("https://")


def test_raw_properties_keeps_everything_unmapped(client, pages):
    """The escape hatch: unmodelled columns still reach the caller."""
    task = client._parse_page(pages[0])
    assert set(task.raw_properties) >= {"Name", "Description", "Status"}


def test_title_joins_every_segment():
    """Notion splits formatted text into segments; reading [0] loses the rest."""
    prop = {"title": [{"plain_text": "Fix "}, {"plain_text": "auth"}]}
    assert NotionClient._plain_text(prop, "title") == "Fix auth"


def test_empty_rich_text_is_none_not_empty_string():
    assert NotionClient._plain_text({"rich_text": []}, "rich_text") is None


def test_missing_property_is_none():
    assert NotionClient._plain_text(None, "title") is None
    assert NotionClient._select_name(None) is None


def test_unset_select_is_none():
    """A row where nobody picked a status sends "select": null."""
    assert NotionClient._select_name({"select": None}) is None
    assert NotionClient._select_name({"select": {"name": "Done"}}) == "Done"


def test_untitled_page_still_parses(client, pages):
    """title is required on NotionTask, but Notion allows untitled rows."""
    page = dict(pages[0])
    page["properties"] = dict(page["properties"], Name={"type": "title", "title": []})
    assert client._parse_page(page).title == "(untitled)"


def test_page_missing_optional_fields_still_parses(client):
    """Only "id" is truly required; everything else degrades to None."""
    task = client._parse_page({"id": "abc", "properties": {}})
    assert task.id == "abc"
    assert task.title == "(untitled)"
    assert (task.description, task.status, task.created_at, task.url) == (
        None,
        None,
        None,
        None,
    )


# --- query filters -------------------------------------------------------


def test_done_rows_are_excluded_by_default():
    """Finished work costs quota and pollutes the ranking, so it never fetches."""
    built = NotionClient._build_filter(None, exclude_done=True)
    conditions = built["or"]
    assert conditions[0]["select"] == {"does_not_equal": "Done"}
    # Notion does not treat an unset select as "not equal to Done", so rows
    # with no status would vanish without this second condition.
    assert conditions[1]["select"] == {"is_empty": True}


def test_an_explicit_status_wins_over_the_done_exclusion():
    """--status Done must return Done rows, not nothing."""
    built = NotionClient._build_filter("Done", exclude_done=True)
    assert built == {"property": "Status", "select": {"equals": "Done"}}


def test_no_filter_when_nothing_is_excluded():
    assert NotionClient._build_filter(None, exclude_done=False) is None
