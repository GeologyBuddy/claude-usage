"""Tests for project-skill attribution: event extraction, scan integration, and
the dashboard's per-run token windows (skill_runs)."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from scanner import extract_skill_event, get_db, scan
import dashboard

NL = chr(10)  # avoid backslash-escaped newline literals in source
PROJECT_SKILL = "C:/work/proj/.claude/skills/ponytail"
OTHER_SKILL = "C:/work/proj/.agents/skills/asd-ste100"
HOME = str(Path.home()).replace(chr(92), "/")


def _skill_body(uuid, base_dir, ts, tool_use_id=None, session_id="s1"):
    rec = {"type": "user", "isMeta": True, "uuid": uuid, "sessionId": session_id,
           "timestamp": ts,
           "message": {"role": "user", "content": [
               {"type": "text", "text": f"Base directory for this skill: {base_dir}{NL}{NL}# Skill{NL}body text"}]}}
    if tool_use_id:
        rec["sourceToolUseID"] = tool_use_id
    return json.dumps(rec)


def _prompt(uuid, text, ts, session_id="s1"):
    return json.dumps({"type": "user", "uuid": uuid, "sessionId": session_id,
                       "timestamp": ts, "message": {"role": "user", "content": text}})


def _tool_result(uuid, ts, session_id="s1"):
    return json.dumps({"type": "user", "uuid": uuid, "sessionId": session_id, "timestamp": ts,
                       "message": {"role": "user", "content": [
                           {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}})


def _assistant(message_id, ts, inp=100, out=10, session_id="s1", sidechain=False):
    return json.dumps({"type": "assistant", "sessionId": session_id, "timestamp": ts,
                       "isSidechain": sidechain, "cwd": "C:/work/proj",
                       "message": {"model": "claude-opus-4-8", "id": message_id,
                                   "usage": {"input_tokens": inp, "output_tokens": out},
                                   "content": []}})


class TestExtractSkillEvent(unittest.TestCase):
    def _event(self, line):
        return extract_skill_event(json.loads(line))

    def test_tool_started_project_skill_is_auto(self):
        e = self._event(_skill_body("u1", PROJECT_SKILL, "2026-04-08T10:00:00Z", tool_use_id="toolu_1"))
        self.assertEqual((e["kind"], e["skill"], e["source"]), ("skill", "ponytail", "auto"))
        self.assertGreater(e["load_chars"], 0)

    def test_typed_project_skill_is_user(self):
        e = self._event(_skill_body("u1", OTHER_SKILL.replace("/", chr(92)), "2026-04-08T10:00:00Z"))
        self.assertEqual((e["skill"], e["source"]), ("asd-ste100", "user"))

    def test_user_level_and_plugin_skills_ignored(self):
        for base in (f"{HOME}/.claude/skills/foo", f"{HOME.upper()}/.agents/skills/foo",
                     f"{HOME}/.claude/plugins/cache/x/1.0/skills/foo"):
            self.assertIsNone(self._event(_skill_body("u1", base, "2026-04-08T10:00:00Z")), base)

    def test_prompt_and_builtin_command_are_prompts(self):
        self.assertEqual(self._event(_prompt("u1", "hello", "t"))["kind"], "prompt")
        builtin = "<command-name>/compact</command-name>"
        self.assertEqual(self._event(_prompt("u2", builtin, "t"))["kind"], "prompt")

    def test_tool_results_and_other_meta_ignored(self):
        self.assertIsNone(self._event(_tool_result("u1", "t")))
        meta = json.loads(_prompt("u2", "<system-reminder>x</system-reminder>", "t"))
        meta["isMeta"] = True
        self.assertIsNone(extract_skill_event(meta))


class TestSkillRunsEndToEnd(unittest.TestCase):
    """One session: ponytail (auto) and asd-ste100 load in one prompt, then a new
    prompt. Each window ends at the next event; subagent turns are excluded."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.projects = self.tmp / "projects"
        self.db_path = self.tmp / "usage.db"
        self.path = self.projects / "proj" / "s1.jsonl"
        self.path.parent.mkdir(parents=True)
        self._write([
            _prompt("p1", "do the thing", "2026-04-08T10:00:00Z"),
            _assistant("m0", "2026-04-08T10:00:01Z", inp=1, out=1),           # before any skill
            _skill_body("k1", PROJECT_SKILL, "2026-04-08T10:00:02Z", tool_use_id="toolu_1"),
            _assistant("m1", "2026-04-08T10:00:05Z", inp=100, out=10),        # ponytail
            _tool_result("r1", "2026-04-08T10:00:06Z"),                       # not a window end
            _assistant("m2", "2026-04-08T10:00:12Z", inp=200, out=20),        # ponytail
            _assistant("ms", "2026-04-08T10:00:13Z", inp=9999, sidechain=True),  # subagent: excluded
            _skill_body("k2", OTHER_SKILL, "2026-04-08T10:00:20Z", tool_use_id="toolu_2"),
            _assistant("m3", "2026-04-08T10:00:30Z", inp=300, out=30),        # asd-ste100
            _prompt("p2", "thanks", "2026-04-08T10:01:00Z"),
            _assistant("m4", "2026-04-08T10:01:05Z", inp=400, out=40),        # no skill
        ])

    def _write(self, lines, mode="w"):
        with open(self.path, mode) as f:
            f.write(NL.join(lines) + NL)

    def _runs(self):
        scan(projects_dir=self.projects, db_path=self.db_path, verbose=False)
        rows = dashboard.get_dashboard_data(self.db_path)["skill_runs"]
        return {r["skill"]: r for r in rows}

    def test_windows_split_between_skills_and_end_at_prompt(self):
        runs = self._runs()
        self.assertEqual(set(runs), {"ponytail", "asd-ste100"})
        p, a = runs["ponytail"], runs["asd-ste100"]
        self.assertEqual((p["turns"], p["input"], p["output"]), (2, 300, 30))
        self.assertEqual((a["turns"], a["input"], a["output"]), (1, 300, 30))
        self.assertEqual(p["duration_ms"], 10000)   # 10:00:02 -> 10:00:12
        self.assertEqual(p["source"], "auto")
        self.assertGreater(p["load_tokens"], 0)

    def test_incremental_rescan_does_not_duplicate_events(self):
        self._runs()
        self._write([_assistant("m5", "2026-04-08T10:02:00Z")], mode="a")
        os.utime(self.path, (1e9, 2e9))
        self._runs()
        conn = get_db(self.db_path)
        n = conn.execute("SELECT COUNT(*) FROM skill_events WHERE kind='skill'").fetchone()[0]
        conn.close()
        self.assertEqual(n, 2)

    def test_backfill_fills_already_processed_files_once(self):
        self._runs()
        conn = get_db(self.db_path)
        conn.execute("DELETE FROM skill_events")
        conn.execute("DELETE FROM schema_meta WHERE key='skill_backfill_done'")
        conn.commit()
        conn.close()
        self.assertEqual(set(self._runs()), {"ponytail", "asd-ste100"})


if __name__ == "__main__":
    unittest.main()
