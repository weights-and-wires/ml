"""The MCP endpoint, its bearer tokens, and the token management API."""

from __future__ import annotations

import importlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SECRET = "test-proxy-secret"
SIGNED_IN = {"Remote-User": "Portal", "X-SSO-Proxy": SECRET}
OTHER = {"Remote-User": "other", "X-SSO-Proxy": SECRET}
ACCEPT = "application/json, text/event-stream"

LESSON = """---
title: Loss and honest scores
slug: 001-alpha
part: "Part 1 — Foundations"
summary: Score a rule on data it has not seen.
minutes: 20
gpu: false
objectives:
  - State one fact.
  - State a second fact.
  - State a third fact.
exercise: 001-alpha
---

# Loss and honest scores

A rule earns a loss on every example, and the risk is their average.

## The empirical risk

The empirical risk averages the loss over the training examples.
"""

SECOND = (
    LESSON.replace("001-alpha", "002-beta")
    .replace("Loss and honest scores", "Curves and bias")
    .replace("The empirical risk averages", "Variance averages")
    .replace("The empirical risk", "Variance")
)

PROJECT = """---
title: Explore before you model
slug: 002p-project-explore
part: "Part 1 — Foundations"
kind: project
summary: Clean a messy export, then fit a line honestly.
minutes: 180
gpu: false
objectives:
  - Load a file.
prereqs:
  - 001-alpha
exercise: 002p-project-explore
---

# Explore before you model

A letting agency sent a messy file of rental listings.
"""

DRAFT = LESSON.replace("001-alpha", "003-hold").replace("---\n\n#", "draft: true\n---\n\n#", 1)

TESTS = """from mlcheck import check, equal


@check("add returns the sum")
def _(sub):
    equal(sub.add(2, 3), 5)
"""


class MCPTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        lessons = self.root / "content" / "lessons"
        lessons.mkdir(parents=True)
        (lessons / "001-alpha.md").write_text(LESSON, encoding="utf-8")
        (lessons / "002-beta.md").write_text(SECOND, encoding="utf-8")
        (lessons / "002p-project-explore.md").write_text(PROJECT, encoding="utf-8")
        (lessons / "003-hold.md").write_text(DRAFT, encoding="utf-8")
        for slug in ("001-alpha", "002-beta", "002p-project-explore", "003-hold"):
            exercise = self.root / "content" / "exercises" / slug
            exercise.mkdir(parents=True)
            (exercise / "meta.json").write_text(
                json.dumps({"id": slug, "title": slug, "brief": "Add two numbers.", "timeout": 30}),
                encoding="utf-8",
            )
            (exercise / "starter.py").write_text("def add(a, b):\n    raise NotImplementedError\n")
            (exercise / "solution.py").write_text("def add(a, b):\n    return a + b\n")
            (exercise / "tests.py").write_text(TESTS)
        self._env = {
            "DATA_DIR": str(self.root / "data"),
            "CONTENT_DIR": str(self.root / "content"),
            "FRONTEND_DIST": str(self.root / "missing-dist"),
            "SSO_PROXY_SECRET": SECRET,
            "SSO_USERS": "portal:reader",
            # A deployed image: the draft lesson and its exercise are not served.
            "GIT_SHA": "0123456789abcdef",
        }
        os.environ.update(self._env)
        from fastapi.testclient import TestClient
        from backend.app import curriculum, db
        from backend.app.config import get_settings
        from backend.app.main import app

        self.get_settings = get_settings
        self.curriculum = curriculum
        self.db = db
        get_settings.cache_clear()
        curriculum.reload_content()
        # Entering the client runs the lifespan, which starts the MCP session manager.
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        for key in self._env:
            os.environ.pop(key, None)
        os.environ.pop("MCP_PUBLIC_URL", None)
        self.get_settings.cache_clear()
        self.curriculum.reload_content()
        self._dir.cleanup()

    # Helpers ------------------------------------------------------------

    def _new_token(self, name: str = "laptop", headers: dict | None = None) -> dict:
        response = self.client.post(
            "/api/mcp/tokens", json={"name": name}, headers=headers or SIGNED_IN
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _rpc(self, token: str | None, method: str, params: dict | None = None,
             path: str = "/mcp", extra: dict | None = None):
        headers = {"Accept": ACCEPT, "Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        headers.update(extra or {})
        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        return self.client.post(path, headers=headers, json=body, follow_redirects=False)

    def _call(self, token: str, tool: str, **arguments):
        response = self._rpc(token, "tools/call", {"name": tool, "arguments": arguments})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()["result"]
        text = result["content"][0]["text"]
        return result.get("isError", False), (text if result.get("isError") else json.loads(text))

    # Token management ---------------------------------------------------

    def test_token_api_needs_sso(self) -> None:
        self.assertEqual(self.client.get("/api/mcp/tokens").status_code, 401)
        self.assertEqual(self.client.post("/api/mcp/tokens", json={"name": "x"}).status_code, 401)
        self.assertEqual(self.client.delete("/api/mcp/tokens/1").status_code, 401)

    def test_create_list_and_revoke(self) -> None:
        listed = self.client.get("/api/mcp/tokens", headers=SIGNED_IN).json()
        self.assertEqual(listed, {"url": "http://testserver/mcp", "tokens": []})

        created = self._new_token()
        self.assertEqual(set(created), {"id", "name", "prefix", "created_at", "token"})
        self.assertTrue(created["token"].startswith("mlmcp_"))
        self.assertEqual(created["prefix"], created["token"][:12])
        self.assertEqual(created["name"], "laptop")

        tokens = self.client.get("/api/mcp/tokens", headers=SIGNED_IN).json()["tokens"]
        self.assertEqual(
            tokens,
            [
                {
                    "id": created["id"],
                    "name": "laptop",
                    "prefix": created["prefix"],
                    "created_at": created["created_at"],
                    "last_used_at": None,
                }
            ],
        )
        # The secret itself is never stored, only its hash.
        with self.db.session() as connection:
            row = dict(connection.execute("SELECT * FROM mcp_tokens").fetchone())
        self.assertNotIn(created["token"], json.dumps(row))
        self.assertEqual(row["username"], "reader")  # after SSO_USERS mapping
        self.assertEqual(len(row["token_hash"]), 64)

        # Another reader neither sees nor revokes it.
        self.assertEqual(self.client.get("/api/mcp/tokens", headers=OTHER).json()["tokens"], [])
        response = self.client.delete(f"/api/mcp/tokens/{created['id']}", headers=OTHER)
        self.assertEqual(response.status_code, 404)

        response = self.client.delete(f"/api/mcp/tokens/{created['id']}", headers=SIGNED_IN)
        self.assertEqual(response.json(), {"revoked": True})
        self.assertEqual(self.client.get("/api/mcp/tokens", headers=SIGNED_IN).json()["tokens"], [])
        response = self.client.delete(f"/api/mcp/tokens/{created['id']}", headers=SIGNED_IN)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self._rpc(created["token"], "ping").status_code, 401)

    def test_public_url_setting(self) -> None:
        os.environ["MCP_PUBLIC_URL"] = "https://ml.ifkash.dev/mcp"
        self.get_settings.cache_clear()
        listed = self.client.get("/api/mcp/tokens", headers=SIGNED_IN).json()
        self.assertEqual(listed["url"], "https://ml.ifkash.dev/mcp")

    def test_name_is_validated(self) -> None:
        for name in ("", "   ", "x" * 65):
            response = self.client.post(
                "/api/mcp/tokens", json={"name": name}, headers=SIGNED_IN
            )
            self.assertEqual(response.status_code, 422, name)

    def test_at_most_ten_active_tokens(self) -> None:
        for i in range(10):
            self._new_token(f"t{i}")
        response = self.client.post("/api/mcp/tokens", json={"name": "t10"}, headers=SIGNED_IN)
        self.assertEqual(response.status_code, 409)

    def test_cli_creates_lists_and_revokes(self) -> None:
        from backend.app import mcp_tokens

        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(mcp_tokens.main(["create", "reader", "--name", "vm"]), 0)
        token = out.getvalue().strip().splitlines()[-1]
        self.assertTrue(token.startswith("mlmcp_"))
        self.assertEqual(self._rpc(token, "ping").status_code, 200)

        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(mcp_tokens.main(["list", "reader"]), 0)
        self.assertIn(token[:12], out.getvalue())
        self.assertNotIn(token, out.getvalue())

        token_id = self.db.list_tokens("reader")[0]["id"]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(mcp_tokens.main(["revoke", str(token_id)]), 0)
        self.assertEqual(self._rpc(token, "ping").status_code, 401)

    # Authentication on /mcp ---------------------------------------------

    def test_mcp_rejects_missing_and_bad_tokens(self) -> None:
        response = self._rpc(None, "ping")
        self.assertEqual(response.status_code, 401)
        self.assertTrue(response.headers["www-authenticate"].startswith("Bearer"))

        response = self._rpc("mlmcp_not-a-real-token", "ping")
        self.assertEqual(response.status_code, 401)
        self.assertIn('error="invalid_token"', response.headers["www-authenticate"])

        response = self._rpc(None, "ping", extra={"Authorization": "Basic cmVhZGVyOng="})
        self.assertEqual(response.status_code, 401)

    def test_mcp_ignores_sso_headers(self) -> None:
        response = self._rpc(None, "ping", extra=SIGNED_IN)
        self.assertEqual(response.status_code, 401)

        # With a valid token, SSO headers can't change who the caller is.
        token = self._new_token(headers=OTHER)["token"]
        _, progress = self._call_with(token, SIGNED_IN, "get_progress")
        self.assertEqual(progress["reader"], "other")

    def test_hub_secret_acts_as_the_named_reader(self) -> None:
        hub = {"X-MCP-Hub-Secret": "hub-secret", "X-MCP-Hub-User": "Portal"}

        # Off by default: without MCP_HUB_SECRET the headers grant nothing.
        self.assertEqual(self._rpc(None, "ping", extra=hub).status_code, 401)

        os.environ["MCP_HUB_SECRET"] = "hub-secret"
        self.get_settings.cache_clear()
        try:
            wrong = {**hub, "X-MCP-Hub-Secret": "guess"}
            self.assertEqual(self._rpc(None, "ping", extra=wrong).status_code, 401)
            no_user = {"X-MCP-Hub-Secret": "hub-secret"}
            self.assertEqual(self._rpc(None, "ping", extra=no_user).status_code, 401)

            response = self._rpc(
                None, "tools/call", {"name": "get_progress", "arguments": {}}, extra=hub
            )
            self.assertEqual(response.status_code, 200, response.text)
            progress = json.loads(response.json()["result"]["content"][0]["text"])
            # The Authelia name maps through SSO_USERS, as a browser session does.
            self.assertEqual(progress["reader"], "reader")
        finally:
            os.environ.pop("MCP_HUB_SECRET", None)
            self.get_settings.cache_clear()

    def _call_with(self, token: str, headers: dict, tool: str, **arguments):
        response = self._rpc(
            token, "tools/call", {"name": tool, "arguments": arguments}, extra=headers
        )
        result = response.json()["result"]
        return result.get("isError", False), json.loads(result["content"][0]["text"])

    # The protocol and the tools -----------------------------------------

    def test_initialize_and_list_tools(self) -> None:
        token = self._new_token()["token"]
        for path in ("/mcp", "/mcp/"):
            response = self._rpc(
                token,
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
                path=path,
            )
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()["result"]
            self.assertEqual(result["protocolVersion"], "2025-06-18")
            self.assertIn("tools", result["capabilities"])

        response = self.client.post(
            "/mcp",
            headers={"Authorization": f"Bearer {token}", "Accept": ACCEPT},
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        self.assertEqual(response.status_code, 202)

        tools = {tool["name"] for tool in self._rpc(token, "tools/list").json()["result"]["tools"]}
        self.assertTrue(
            {
                "course_overview", "list_lessons", "get_lesson", "search", "list_projects",
                "get_project", "list_exercises", "get_exercise", "get_solution",
                "get_document", "get_book", "get_progress", "list_runs", "get_compute",
                "get_draft", "save_draft", "reset_draft", "set_progress", "export_my_data",
                "run_exercise",
            }
            <= tools
        )
        templates = self._rpc(token, "resources/templates/list").json()["result"]
        uris = [t["uriTemplate"] for t in templates["resourceTemplates"]]
        self.assertIn("ml://lessons/{slug}", uris)
        self.assertIn("ml://exercises/{exercise_id}", uris)

    def test_last_used_is_recorded(self) -> None:
        token = self._new_token()["token"]
        self.assertEqual(self._rpc(token, "ping").status_code, 200)
        tokens = self.client.get("/api/mcp/tokens", headers=SIGNED_IN).json()["tokens"]
        self.assertIsNotNone(tokens[0]["last_used_at"])

    def test_get_lesson_does_not_mark_progress(self) -> None:
        token = self._new_token()["token"]
        error, lesson = self._call(token, "get_lesson", slug="001-alpha")
        self.assertFalse(error)
        self.assertIn("empirical risk", lesson["body"])
        self.assertEqual(lesson["next"], "002-beta")
        self.assertEqual(lesson["exercise"], "001-alpha")
        self.assertIsNone(lesson["status"])
        self.assertEqual(self.db.get_progress("reader"), {})

        error, message = self._call(token, "get_lesson", slug="nope")
        self.assertTrue(error)
        self.assertIn("No lesson", message)

        response = self._rpc(token, "resources/read", {"uri": "ml://lessons/002-beta"})
        self.assertIn("Variance", response.json()["result"]["contents"][0]["text"])

    def test_search(self) -> None:
        token = self._new_token()["token"]
        error, found = self._call(token, "search", query="EMPIRICAL risk")
        self.assertFalse(error)
        chapters = [r for r in found["results"] if r["kind"] == "chapter"]
        self.assertEqual([r["slug"] for r in chapters], ["001-alpha"])
        hit = chapters[0]
        self.assertEqual(hit["section"], "The empirical risk")
        self.assertIn("empirical risk", hit["snippet"].lower())

        _, found = self._call(token, "search", query="variance averages", limit=1)
        self.assertEqual(found["results"][0]["slug"], "002-beta")
        self.assertEqual(len(found["results"]), 1)

        # Projects, exercises, and documents are searched too.
        _, found = self._call(token, "search", query="letting agency")
        self.assertEqual([(r["kind"], r["slug"]) for r in found["results"]],
                         [("project", "002p-project-explore")])
        _, found = self._call(token, "search", query="add two numbers")
        self.assertIn(("exercise", "001-alpha"), [(r["kind"], r["slug"]) for r in found["results"]])
        _, found = self._call(token, "search", query="curriculum")
        self.assertIn(("document", "CURRICULUM.md"),
                      [(r["kind"], r["slug"]) for r in found["results"]])

        error, message = self._call(token, "search", query="   ")
        self.assertTrue(error)

    def test_overview_exercise_and_documents(self) -> None:
        token = self._new_token()["token"]
        _, overview = self._call(token, "course_overview")
        self.assertEqual(overview["reader"], "reader")
        self.assertEqual(
            overview["counts"], {"parts": 2, "chapters": 2, "projects": 1, "exercises": 3}
        )
        self.assertEqual(overview["parts"][0]["lessons"][0]["slug"], "001-alpha")
        self.assertEqual(overview["parts"][0]["lessons"][0]["number"], "1")
        # Projects close the list, apart from the chapters.
        self.assertEqual(overview["parts"][-1]["part"], "Projects")
        self.assertEqual(overview["parts"][-1]["lessons"][0]["number"], "P1")
        self.assertIsNone(overview["book"])  # no book has been printed here

        _, exercise = self._call(token, "get_exercise", exercise_id="001-alpha")
        self.assertIn("NotImplementedError", exercise["starter"])
        self.assertIn("@check", exercise["tests"])
        self.assertNotIn("solution", exercise)
        self.assertEqual(exercise["lessons"], ["001-alpha"])
        _, solution = self._call(token, "get_solution", exercise_id="001-alpha")
        self.assertIn("return a + b", solution["solution"])

        error, _ = self._call(token, "get_exercise", exercise_id="003-hold")
        self.assertTrue(error)  # the exercise of a draft lesson is not served

        _, document = self._call(token, "get_document", name="curriculum")
        self.assertEqual(document["name"], "CURRICULUM.md")
        error, _ = self._call(token, "get_document", name="../README")
        self.assertTrue(error)

    def test_lists_cover_chapters_projects_and_exercises(self) -> None:
        token = self._new_token()["token"]
        _, listed = self._call(token, "list_lessons")
        self.assertEqual(
            [lesson["slug"] for lesson in listed["lessons"]],
            ["001-alpha", "002-beta", "002p-project-explore"],  # the draft is hidden
        )
        _, listed = self._call(token, "list_lessons", kind="project")
        self.assertEqual([lesson["slug"] for lesson in listed["lessons"]], ["002p-project-explore"])
        _, listed = self._call(token, "list_lessons", part="foundations", kind="chapter")
        self.assertEqual(listed["count"], 2)
        self.assertEqual(listed["lessons"][0]["headings"][0]["text"], "The empirical risk")

        _, projects = self._call(token, "list_projects")
        project = projects["projects"][0]
        self.assertEqual(project["number"], "P1")
        self.assertEqual(project["after_chapter"], 2)
        self.assertEqual(project["prereqs"], ["001-alpha"])
        _, found = self._call(token, "get_project", slug="002p-project-explore")
        self.assertIn("letting agency", found["body"])
        self.assertEqual(found["after_chapter"], 2)
        error, _ = self._call(token, "get_project", slug="001-alpha")  # a chapter
        self.assertTrue(error)
        # A project reads through get_lesson, like any lesson.
        _, lesson = self._call(token, "get_lesson", slug="002p-project-explore")
        self.assertEqual(lesson["kind"], "project")
        self.assertEqual(lesson["index"], 1)

        self.db.set_progress("reader", "001-alpha", "solved")
        _, exercises = self._call(token, "list_exercises")
        self.assertEqual(exercises["count"], 3)
        by_id = {row["id"]: row for row in exercises["exercises"]}
        self.assertEqual(by_id["001-alpha"]["status"], "solved")
        self.assertEqual(by_id["001-alpha"]["lessons"], ["001-alpha"])
        self.assertIsNone(by_id["002-beta"]["status"])
        self.assertNotIn("003-hold", by_id)

    def test_book_and_compute(self) -> None:
        token = self._new_token()["token"]
        _, book = self._call(token, "get_book")
        self.assertEqual(book, {"book": None})
        _, compute = self._call(token, "get_compute")
        self.assertEqual(compute["runpod"], {"configured": False})
        self.assertIn("workers", compute["local"])

    def test_drafts_progress_and_runs_act_as_the_token_owner(self) -> None:
        token = self._new_token()["token"]
        _, saved = self._call(token, "save_draft", exercise_id="001-alpha", code="x = 1\n")
        self.assertTrue(saved["saved"])
        self.assertEqual(self.db.get_draft("reader", "001-alpha"), "x = 1\n")
        _, draft = self._call(token, "get_draft", exercise_id="001-alpha")
        self.assertEqual(draft["code"], "x = 1\n")

        _, marked = self._call(token, "set_progress", slug="002-beta", status="started")
        self.assertEqual(marked["status"], "started")

        code = "def add(a, b):\n    print('adding')\n    return a + b\n"
        error, result = self._call(token, "run_exercise", exercise_id="001-alpha", code=code)
        self.assertFalse(error, result)
        self.assertTrue(result["ok"], result)
        self.assertEqual((result["passed"], result["total"]), (1, 1))
        self.assertEqual(result["checks"][0]["status"], "passed")
        self.assertEqual(result["lessons_solved"], ["001-alpha"])
        self.assertEqual(self.db.get_progress("reader")["001-alpha"]["status"], "solved")
        self.assertEqual(self.db.get_draft("reader", "001-alpha"), code)

        _, progress = self._call(token, "get_progress")
        self.assertEqual(progress["summary"]["done"], 1)
        self.assertEqual(progress["runs"][0]["exercise_id"], "001-alpha")
        self.assertEqual(
            [(row["slug"], row["status"]) for row in progress["lessons"]],
            [("001-alpha", "solved"), ("002-beta", "started")],
        )

        error, result = self._call(
            token, "run_exercise", exercise_id="002-beta", code="print('hi')\n", mode="run"
        )
        self.assertFalse(error, result)
        self.assertIn("hi", result["stdout"])
        self.assertEqual(self.db.get_progress("reader")["002-beta"]["status"], "started")

        _, runs = self._call(token, "list_runs")
        self.assertEqual([(r["exercise_id"], r["mode"]) for r in runs["runs"]],
                         [("002-beta", "run"), ("001-alpha", "check")])
        _, runs = self._call(token, "list_runs", exercise_id="001-alpha", limit=1)
        self.assertEqual(len(runs["runs"]), 1)
        self.assertEqual(runs["runs"][0]["ok"], 1)

        _, exported = self._call(token, "export_my_data")
        self.assertEqual(set(exported), {"progress", "drafts", "runs"})
        self.assertEqual(len(exported["runs"]), 2)

        _, reset = self._call(token, "reset_draft", exercise_id="001-alpha")
        self.assertFalse(reset["saved"])
        self.assertIsNone(self.db.get_draft("reader", "001-alpha"))
        # The progress that the draft's run earned stays.
        self.assertEqual(self.db.get_progress("reader")["001-alpha"]["status"], "solved")

    def test_tools_are_scoped_to_the_token_owner(self) -> None:
        mine = self._new_token()["token"]
        other = self._new_token(headers=OTHER)["token"]
        self._call(mine, "save_draft", exercise_id="001-alpha", code="x = 1\n")
        _, draft = self._call(other, "get_draft", exercise_id="001-alpha")
        self.assertIsNone(draft["code"])
        _, progress = self._call(other, "get_progress")
        self.assertEqual(progress["reader"], "other")
        self.assertEqual(progress["lessons"], [])


class SpaFallbackTests(unittest.TestCase):
    """The single-page app's catch-all must not answer for API, MCP, or OAuth paths."""

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        root = Path(self._dir.name)
        (root / "dist").mkdir()
        (root / "dist" / "index.html").write_text("<!doctype html><title>app</title>")
        self._saved = {k: os.environ.get(k) for k in ("DATA_DIR", "FRONTEND_DIST")}
        os.environ["DATA_DIR"] = str(root / "data")
        os.environ["FRONTEND_DIST"] = str(root / "dist")
        from backend.app import main
        from backend.app.config import get_settings

        self.main, self.get_settings = main, get_settings
        get_settings.cache_clear()
        # The catch-all is mounted when the module loads, so load it again
        # with a dist folder. tearDown loads it once more without one.
        importlib.reload(main)

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.get_settings.cache_clear()
        importlib.reload(self.main)
        self._dir.cleanup()

    def test_reserved_prefixes_return_json_404(self) -> None:
        from fastapi.testclient import TestClient

        with TestClient(self.main.app) as client:
            for path in (
                "/api/nope",
                "/mcp/extra",
                "/.well-known/oauth-authorization-server",
                "/.well-known/oauth-protected-resource/mcp",
            ):
                response = client.get(path)
                self.assertEqual(response.status_code, 404, path)
                self.assertEqual(response.json(), {"detail": "Not found"}, path)
            response = client.get("/lessons/anything")
            self.assertEqual(response.status_code, 200)
            self.assertIn("<title>app</title>", response.text)


if __name__ == "__main__":
    unittest.main()
