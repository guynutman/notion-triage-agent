# Notion Task Triage Agent

[![CI](https://github.com/guynutman/notion-triage-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/guynutman/notion-triage-agent/actions/workflows/ci.yml)

An LLM agent that reads your Notion task database, classifies every item, extracts
concrete action items, assigns a priority, and prints a ranked list of what to work
on next.

Built with **LangGraph** for orchestration and **Pydantic v2** for structured output —
every model call returns a validated object, never raw text.

```
📋 Triage Results — 5 tasks analyzed

 1. 🔴 [CRITICAL] Fix auth redirect loop
    Category: task (0.95 confidence)
    Why: Blocks three other tasks and has a due date this week.
    Actions:
      • Debug the OAuth callback handler (~30 min)
      • Add error logging to the auth middleware (~15 min)

 2. 🟡 [MEDIUM] Read up on vector databases
    Category: reference (0.82 confidence)
    Why: Useful background, but nothing depends on it yet.
    Actions:
      • Compare Pinecone and Weaviate pricing (~20 min)

💡 Recommendation:
   Start with the auth redirect loop — it is blocking other work and has clearly
   scoped action items. The vector database reading can wait for a gap.
   Total estimated: ~65 min
```

## How it works

```
                   ┌──────────────┐
   Notion DB ─────▶│ fetch_tasks  │  1 HTTP request (paginated)
                   └──────┬───────┘
                          │
                 nothing fetched? ────────────────▶ END
                          ▼
                   ┌──────────────┐
                   │ analyze_tasks│  1 LLM call per task, run in parallel
                   └──────┬───────┘  → TaskAnalysis
                          ▼
                   ┌──────────────┐
                   │  recommend   │  1 LLM call → Recommendation
                   └──────┬───────┘
                          ▼
                   ranked CLI report
                          │
                   --write-back? ───▶ ┌────────────┐
                                      │ write_back │ → back into Notion
                                      └────────────┘
```

Each step is a LangGraph node: a plain function that takes the pipeline state and
returns only the fields it changed. The framework merges those updates.

## Setup

**Requirements:** Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/guynutman/notion-triage-agent.git
cd notion-triage-agent
uv sync
```

### 1. Create a Notion integration

Go to [notion.so/profile/integrations](https://www.notion.so/profile/integrations) →
**New integration** → type **Internal**, capabilities **Read content** and
**Update content**. Copy the Internal Integration Secret (`ntn_…`).

### 2. Create a task database

A full-page Notion database with these columns:

| Column        | Type      |
| ------------- | --------- |
| `Name`        | Title     |
| `Description` | Text      |
| `Status`      | Select    |

Then **share it with the integration**: on the database page, `•••` → **Connections**
→ pick your integration. Skipping this returns zero rows with no error — it is the
single most common setup mistake.

### 3. Get the database ID

From the database URL, the 32 hex characters before the `?`:

```
notion.so/workspace/Task-Inbox-1a2b3c4d5e6f7890abcdef1234567890?v=…
                                └──────── database ID ────────┘
```

### 4. Get a Gemini API key

From [aistudio.google.com/apikey](https://aistudio.google.com/apikey). The free tier
is enough for this project.

### 5. Configure

Create `.env` in the project root:

```
NOTION_TOKEN=ntn_…
NOTION_DATABASE_ID=1a2b3c…
GEMINI_API_KEY=…
```

`.env` is gitignored.

## Usage

```bash
uv run notion-triage-agent                          # triage everything
uv run notion-triage-agent --status "Not started"   # only unstarted rows
uv run notion-triage-agent --limit 10               # cap the number of tasks
uv run notion-triage-agent --write-back             # push results back to Notion
uv run notion-triage-agent --workers 8              # more parallel model calls
```

| Flag           | Effect                                                                  |
| -------------- | ----------------------------------------------------------------------- |
| `--status`     | Server-side filter on the `Status` column, so unwanted rows never fetch  |
| `--limit N`    | Stop after N tasks; also shrinks the requested page size                 |
| `--workers N`  | Parallel model calls during analysis (default 4; `1` forces sequential)  |
| `--write-back` | Writes each task's category and priority into Notion                     |

### Write-back

`--write-back` adds two select columns to your database, **`AI Category`** and
**`AI Priority`**, and sets them on every analyzed row. The columns are created
automatically on first use — the integration needs the *Update content* capability
for this to work. Nothing else in your database is modified, and a page that fails
to update is reported without stopping the rest.

## Architecture

Six modules, each with one job. Dependencies point inward — nothing imports sideways.

| Module             | Responsibility                                        | Knows about          |
| ------------------ | ----------------------------------------------------- | -------------------- |
| `models.py`        | Every data shape: API rows, LLM outputs, graph state   | nothing              |
| `notion_client.py` | HTTP, pagination, Notion property parsing              | Notion API           |
| `llm.py`           | The `LLMClient` protocol and its Gemini implementation | the model vendor     |
| `nodes.py`         | The triage logic, one function per pipeline step       | neither of the above |
| `graph.py`         | Wiring nodes into a `StateGraph`                       | LangGraph            |
| `cli.py`           | Env vars, dependency construction, output formatting   | everything, thinly   |

Four decisions worth explaining:

**Structured output everywhere.** Every model call passes a Pydantic schema and
returns a validated instance. `confidence` is bounded to 0–1, `category` and
`priority` are enums — a malformed generation is rejected at the boundary that
produced it, not three steps downstream. The `Field(description=…)` text on those
models is emitted into the JSON schema the model receives, so the schema doubles as
part of the prompt.

**Clients are injected, never imported.** Nodes take their dependencies as
keyword-only arguments; `graph.py` binds the real ones with `functools.partial`, so
LangGraph only ever passes state. The payoff is the test suite: every node, and the
entire compiled graph, runs against fakes with no network and no API keys — which is
also why CI needs no secrets.

**The vendor SDK lives in one file.** `nodes.py` depends on a two-method `Protocol`,
not on `google.genai`. Swapping providers means adding a class to `llm.py` and
changing one line in `cli.py`; no data structures and no logic change.

**Model output is not trusted.** The model is asked to echo each task's ID, and the
returned value is overwritten with the real one regardless. Task IDs in the final
ranking are reconciled against the actual analyses — invented IDs are dropped and
omitted ones appended — so the report can never reference a task that does not exist.

### Concurrency

`analyze_tasks` submits one model call per task to a `ThreadPoolExecutor`. The work
is network-bound, so threads are sufficient and asyncio would add no throughput.
Results are collected in *submission* order rather than completion order, so the
report is identical no matter which response arrives first.

### Failure behaviour

Errors accumulate in state instead of aborting the run. A model call that returns
unparseable output is retried once; if it fails again, that task is skipped, the
error is recorded, and the remaining tasks are still analyzed. A partially failed run
prints its results with the failures listed underneath.

This relies on a LangGraph detail: node updates *replace* a state field by default,
so `errors` is declared as `Annotated[list[str], operator.add]` to make updates
append instead.

## Testing

```bash
uv run pytest          # 49 tests, no network, no API keys
uv run ruff check .
```

| Suite                   | Covers                                                             |
| ----------------------- | ------------------------------------------------------------------ |
| `test_models.py`        | Range and enum validation, nested parsing, empty-state construction |
| `test_notion_client.py` | Property parsing against a saved API response fixture               |
| `test_nodes.py`         | Retry, per-task failure isolation, ID reconciliation, prompt inputs, parallel ordering, write-back |
| `test_graph.py`         | The compiled pipeline end to end with fakes, conditional routing, run options |
| `test_cli.py`           | Argument parsing, config validation, report formatting             |

CI runs lint, format check, and the full suite on Python 3.12 and 3.13.

## Tech

Python 3.12 · LangGraph · Pydantic v2 · httpx · Google Gemini API · uv · pytest · ruff
