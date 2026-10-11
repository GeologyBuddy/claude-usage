"""Tests for the GitHub Copilot Chat reader: change-log replay, request rows,
incremental scans, and the dashboard's copilot_requests payload."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import dashboard
from scanner import get_db, init_db, parse_copilot_chat, scan, scan_copilot

NL = chr(10)


def _request(rid, ts, model="copilot/auto"):
    return {"requestId": rid, "timestamp": ts, "modelId": model,
            "message": {"text": f"question {rid}"}, "response": []}


# A chat log in VS Code's patch format: kind 0 snapshot, kind 1 set, kind 2
# append (with "i" = cut the array to that index first).
CHAT = [
    {"kind": 0, "v": {"sessionId": "chat-1", "requests": []}},
    {"kind": 2, "k": ["requests"], "v": [_request("r1", 1789953864678)]},
    {"kind": 2, "k": ["requests", 0, "response"], "v": ["a", "b", "c"]},
    {"kind": 2, "k": ["requests", 0, "response"], "i": 1, "v": ["B"]},  # -> ["a", "B"]
    {"kind": 1, "k": ["requests", 0, "promptTokens"], "v": 18740},
    {"kind": 1, "k": ["requests", 0, "completionTokens"], "v": 1118},
    {"kind": 1, "k": ["requests", 0, "copilotCredits"], "v": 0.7},
    {"kind": 1, "k": ["requests", 0, "elapsedMs"], "v": 14171},
    {"kind": 1, "k": ["requests", 0, "result"], "v": {"metadata": {
        "resolvedModel": "gpt-6-luna", "toolCallRounds": [{}, {}, {}]}}},
    {"kind": 2, "k": ["requests"], "v": [_request("r2", 1789953900000, "copilot/claude-sonnet-5")]},
    {"kind": 1, "k": ["requests", 1, "completionTokens"], "v": 50},
    {"kind": 2, "k": ["requests"], "v": [_request("r3", 1789953999000)]},  # cancelled: no usage
    {"kind": 1, "k": ["customTitle"], "v": "Fix the parser"},
]


class CopilotTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.user = self.tmp / "Code" / "User"
        ws = self.user / "workspaceStorage" / "abc123"
        (ws / "chatSessions").mkdir(parents=True)
        (ws / "workspace.json").write_text(json.dumps({"folder": "file:///c%3A/Users/me/VS/Dev.Bud"}))
        self.chat = ws / "chatSessions" / "chat-1.jsonl"
        self._write(CHAT)
        self.db_path = self.tmp / "usage.db"

    def _write(self, records):
        self.chat.write_text(NL.join(json.dumps(r) for r in records) + NL, encoding="utf-8")

    def _rows(self):
        conn = get_db(self.db_path)
        rows = {r["request_id"]: dict(r) for r in conn.execute("SELECT * FROM copilot_requests")}
        conn.close()
        return rows


class TestParseCopilotChat(CopilotTestBase):
    def test_replay_builds_request_rows(self):
        session_id, rows = parse_copilot_chat(self.chat)
        self.assertEqual(session_id, "chat-1")
        self.assertEqual([r["request_id"] for r in rows], ["r1", "r2"])  # r3 cancelled
        r1, r2 = rows
        self.assertEqual((r1["model"], r1["prompt_tokens"], r1["output_tokens"], r1["credits"],
                          r1["elapsed_ms"], r1["rounds"]), ("gpt-6-luna", 18740, 1118, 0.7, 14171, 3))
        self.assertEqual(r1["timestamp"], "2026-09-21T01:24:24.678Z")
        self.assertEqual((r2["model"], r2["prompt_tokens"], r2["credits"]), ("claude-sonnet-5", 0, 0.0))
        self.assertEqual((r1["title"], r1["workspace"]), ("Fix the parser", "VS/Dev.Bud"))

    def test_untitled_chat_falls_back_to_first_question(self):
        self._write(CHAT[:-1])
        self.assertEqual(parse_copilot_chat(self.chat)[1][0]["title"], "question r1")

    def test_chat_without_folder(self):
        empty = self.user / "globalStorage" / "emptyWindowChatSessions" / "e.jsonl"
        empty.parent.mkdir(parents=True)
        empty.write_text(self.chat.read_text(encoding="utf-8"), encoding="utf-8")
        self.assertEqual(parse_copilot_chat(empty)[1][0]["workspace"], "(no folder)")


class TestScanCopilot(CopilotTestBase):
    def test_scan_then_skip_then_replace_on_change(self):
        conn = get_db(self.db_path)
        init_db(conn)
        self.assertEqual(scan_copilot(conn, [self.user]), 1)
        self.assertEqual(scan_copilot(conn, [self.user]), 0)  # unchanged -> skipped
        # Request r2 is removed from the chat (e.g. undone): the rescan reflects it.
        self._write(CHAT[:9])
        os.utime(self.chat, (1e9, 2e9))
        self.assertEqual(scan_copilot(conn, [self.user]), 1)
        conn.close()
        self.assertEqual(set(self._rows()), {"r1"})

    def test_broken_file_is_skipped_not_fatal(self):
        self.chat.write_text('{"kind": 1, "k": ["missing", "x"], "v": 1}' + NL, encoding="utf-8")
        conn = get_db(self.db_path)
        init_db(conn)
        self.assertEqual(scan_copilot(conn, [self.user]), 0)
        conn.close()

    def test_scan_with_custom_projects_dir_skips_copilot_unless_asked(self):
        projects = self.tmp / "projects"
        projects.mkdir()
        scan(projects_dir=projects, db_path=self.db_path, verbose=False)
        self.assertEqual(self._rows(), {})
        result = scan(projects_dir=projects, db_path=self.db_path, verbose=False, copilot_dirs=[self.user])
        self.assertEqual(result["copilot"], 1)
        self.assertEqual(set(self._rows()), {"r1", "r2"})

    def test_dashboard_payload(self):
        scan(projects_dir=self.tmp / "none", db_path=self.db_path, verbose=False, copilot_dirs=[self.user])
        rows = dashboard.get_dashboard_data(self.db_path)["copilot_requests"]
        self.assertEqual([(r["model"], r["day"], r["output"]) for r in rows],
                         [("gpt-6-luna", "2026-09-21", 1118), ("claude-sonnet-5", "2026-09-21", 50)])


if __name__ == "__main__":
    unittest.main()
