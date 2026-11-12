"""Module-boundary tests.

The layering is a design claim, so it is enforced here rather than trusted:
models.py is the shared core, and the three peer modules (the two service
adapters and the node logic) never import one another.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parent.parent / "src" / "notion_triage_agent"
PEERS = {"llm", "notion_client", "nodes"}


def project_imports(module: str) -> set[str]:
    tree = ast.parse((PACKAGE / f"{module}.py").read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "notion_triage_agent":
                found.update(alias.name for alias in node.names)
            elif node.module.startswith("notion_triage_agent."):
                found.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("notion_triage_agent."):
                    found.add(alias.name.split(".")[1])
    return found


def test_models_imports_nothing_from_the_project():
    assert project_imports("models") == set()


@pytest.mark.parametrize("module", sorted(PEERS))
def test_peer_modules_import_only_the_shared_core(module):
    assert project_imports(module) <= {"models"}


def test_graph_sees_protocols_not_adapters():
    assert project_imports("graph") <= {"models", "nodes"}
