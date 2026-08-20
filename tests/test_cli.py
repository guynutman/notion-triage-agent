"""CLI tests: config loading and report formatting.

format_results is pure -- state in, string out -- so the whole report is
testable without running the graph.
"""

import pytest

from notion_triage_agent.cli import format_results, load_config, parse_args
from notion_triage_agent.models import AgentState, Recommendation
from tests.test_nodes import make_analysis, make_task


def test_load_config_names_every_missing_variable_at_once(monkeypatch):
    """One run should tell you everything to fix, not just the first thing."""
    monkeypatch.setattr("notion_triage_agent.cli.load_dotenv", lambda: None)
    for name in ("NOTION_TOKEN", "NOTION_DATABASE_ID", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        load_config()

    message = str(excinfo.value)
    assert all(name in message for name in ("NOTION_TOKEN", "NOTION_DATABASE_ID", "GEMINI_API_KEY"))


def test_load_config_returns_the_values(monkeypatch):
    monkeypatch.setattr("notion_triage_agent.cli.load_dotenv", lambda: None)
    monkeypatch.setenv("NOTION_TOKEN", "ntn_x")
    monkeypatch.setenv("NOTION_DATABASE_ID", "db")
    monkeypatch.setenv("GEMINI_API_KEY", "key")

    assert load_config() == {
        "NOTION_TOKEN": "ntn_x",
        "NOTION_DATABASE_ID": "db",
        "GEMINI_API_KEY": "key",
    }


def test_empty_state_reports_nothing_to_do():
    assert "No tasks to triage." in format_results(AgentState())


def test_report_follows_the_recommended_order():
    """The ranking, not the fetch order, decides what prints first."""
    state = AgentState(
        raw_tasks=[make_task("t1", "First fetched"), make_task("t2", "Second fetched")],
        analyses=[make_analysis("t1"), make_analysis("t2")],
        recommendation=Recommendation(ranked_tasks=["t2", "t1"], reasoning="why"),
    )

    report = format_results(state)

    assert report.index("Second fetched") < report.index("First fetched")
    assert "2 tasks analyzed" in report
    assert "why" in report


def test_report_renders_priority_category_and_actions():
    state = AgentState(
        raw_tasks=[make_task("t1", "Fix auth")],
        analyses=[make_analysis("t1")],
        recommendation=Recommendation(ranked_tasks=["t1"], reasoning="r"),
    )

    report = format_results(state)

    assert "[HIGH] Fix auth" in report
    assert "task (0.90 confidence)" in report
    assert "• do it (~30 min)" in report


def test_errors_surface_in_the_report():
    """A partially failed run still prints results, with the failures listed."""
    state = AgentState(
        raw_tasks=[make_task("t1")],
        analyses=[make_analysis("t1")],
        errors=["analyze_tasks[Other]: quota exceeded"],
    )

    report = format_results(state)

    assert "1 error(s)" in report
    assert "quota exceeded" in report


# --- argument parsing ----------------------------------------------------


def test_default_arguments():
    args = parse_args([])
    assert args.status is None
    assert args.limit is None
    assert args.write_back is False
    assert args.workers >= 1


def test_arguments_are_parsed():
    args = parse_args(["--status", "Not started", "--limit", "3", "--write-back"])
    assert args.status == "Not started"
    assert args.limit == 3
    assert args.write_back is True


@pytest.mark.parametrize("argv", [["--limit", "0"], ["--workers", "0"]])
def test_nonsensical_counts_are_rejected(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)
