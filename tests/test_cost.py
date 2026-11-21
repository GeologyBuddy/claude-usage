"""Tests for cost consistency: per-model session cost, 1-hour cache writes,
the Haiku 5.5 long-prompt tier, and CLI / dashboard-JS agreement."""

import contextlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cli
import dashboard
from cli import calc_cost, get_pricing
from scanner import (get_db, init_db, insert_turns, parse_jsonl_file, scan,
                     upsert_sessions, _cache_1h_tokens)


def _turn(message_id, model, inp=0, out=0, cr=0, cc=0, cc1h=0, is_subagent=0, agent_id=None):
    return {
        "session_id": "sess-1", "timestamp": "2026-11-21T10:00:00Z", "model": model,
        "input_tokens": inp, "output_tokens": out,
        "cache_read_tokens": cr, "cache_creation_tokens": cc,
        "cache_creation_1h_tokens": cc1h,
        "tool_name": None, "cwd": "/home/user/proj",
        "message_id": message_id, "is_subagent": is_subagent, "agent_id": agent_id,
    }


# A Fable main thread with Opus and Haiku subagents: the case that was priced
# wholly at the Fable rate.
MIXED_TURNS = [
    _turn("m1", "claude-fable-5-2", inp=100, out=70_000, cr=10_000_000, cc=400_000, cc1h=400_000),
    _turn("m2", "claude-fable-5-2", inp=50, out=1_000, cr=500_000, cc=5_000, cc1h=0),
    _turn("s1", "claude-opus-5-5", inp=120, out=160_000, cr=11_000_000, cc=900_000,
          is_subagent=1, agent_id="agent-1"),
    # Haiku 5.5, prompt over 100K tokens (20 + 275,000 + 50,000): the 5x tier.
    _turn("s2", "claude-haiku-5-5", inp=20, out=7_000, cr=275_000, cc=50_000, cc1h=30_000,
          is_subagent=1, agent_id="agent-2"),
    # Same dispatch, prompt of exactly 100K tokens: the base tier.
    _turn("s3", "claude-haiku-5-5", inp=10_000, out=2_000, cr=80_000, cc=10_000,
          is_subagent=1, agent_id="agent-2"),
]
LONG_TURN = MIXED_TURNS[3]


def _turn_cost(t):
    """Cost of one turn, priced on its own: the reference for every aggregate."""
    cost = calc_cost(t["model"], t["input_tokens"], t["output_tokens"],
                     t["cache_read_tokens"], t["cache_creation_tokens"],
                     t["cache_creation_1h_tokens"])
    prompt = t["input_tokens"] + t["cache_read_tokens"] + t["cache_creation_tokens"]
    return cost * 5 if t["model"].startswith("claude-haiku-5-5") and prompt > 100_000 else cost


def _expected_cost(turns):
    return sum(_turn_cost(t) for t in turns)


def _total(key):
    return sum(t[key] for t in MIXED_TURNS)


def _mixed_db():
    db_path = Path(tempfile.mkdtemp()) / "usage.db"
    conn = get_db(db_path)
    init_db(conn)
    upsert_sessions(conn, [{
        "session_id": "sess-1", "project_name": "user/proj",
        "first_timestamp": "2026-11-21T10:00:00Z", "last_timestamp": "2026-11-21T11:00:00Z",
        "git_branch": "main", "model": "claude-fable-5-2",
        "total_input_tokens": _total("input_tokens"),
        "total_output_tokens": _total("output_tokens"),
        "total_cache_read": _total("cache_read_tokens"),
        "total_cache_creation": _total("cache_creation_tokens"),
        "turn_count": len(MIXED_TURNS),
    }])
    insert_turns(conn, MIXED_TURNS)
    conn.commit()
    conn.close()
    return db_path


class TestPricingLookup(unittest.TestCase):
    def test_fable_5_2_has_its_own_price(self):
        for model in ("claude-fable-5-2", "claude-mythos-5-2", "claude-fable-5-2-20261101"):
            p = get_pricing(model)
            self.assertEqual((p["input"], p["output"], p["cache_read"], p["cache_write"]),
                             (8.00, 40.00, 0.20, 10.00), model)

    def test_date_suffix_is_not_read_as_a_minor_version(self):
        """claude-fable-5-20260601 is Fable 5, not claude-fable-5-2."""
        self.assertEqual(get_pricing("claude-fable-5-20260601")["input"], 10.00)
        self.assertEqual(get_pricing("claude-fable-5-10000000")["cache_read"], 1.00)


class TestCacheWrite1h(unittest.TestCase):
    def test_one_hour_write_costs_twice_the_input_rate(self):
        # Opus 5.5: input $4, so 1h write $8/MTok and 5m write $5/MTok.
        self.assertAlmostEqual(calc_cost("claude-opus-5-5", 0, 0, 0, 1_000_000, 1_000_000), 8.00)
        self.assertAlmostEqual(calc_cost("claude-opus-5-5", 0, 0, 0, 1_000_000, 0), 5.00)
        self.assertAlmostEqual(calc_cost("claude-opus-5-5", 0, 0, 0, 1_000_000, 250_000), 5.75)

    def test_default_is_the_five_minute_rate(self):
        self.assertAlmostEqual(calc_cost("claude-opus-5-5", 0, 0, 0, 1_000_000), 5.00)

    def test_usage_block_split(self):
        usage = {"cache_creation_input_tokens": 300,
                 "cache_creation": {"ephemeral_1h_input_tokens": 200, "ephemeral_5m_input_tokens": 100}}
        self.assertEqual(_cache_1h_tokens(usage), 200)
        self.assertEqual(_cache_1h_tokens({"cache_creation_input_tokens": 300}), 0)
        # Never more than the total, so the 5-minute part cannot go negative.
        self.assertEqual(_cache_1h_tokens({"cache_creation_input_tokens": 50,
                                           "cache_creation": {"ephemeral_1h_input_tokens": 200}}), 50)


def _record(message_id, one_hour):
    return json.dumps({
        "type": "assistant", "sessionId": "sess-1", "timestamp": "2026-11-21T10:00:00Z",
        "cwd": "/home/user/proj",
        "message": {"id": message_id, "model": "claude-fable-5-2", "content": [],
                    "usage": {"input_tokens": 1, "output_tokens": 2,
                              "cache_read_input_tokens": 3, "cache_creation_input_tokens": 500,
                              "cache_creation": {"ephemeral_1h_input_tokens": one_hour,
                                                 "ephemeral_5m_input_tokens": 500 - one_hour}}},
    })


class TestScannerCache1h(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.projects = self.tmp / "projects"
        self.projects.mkdir()
        self.transcript = self.projects / "a.jsonl"
        self.transcript.write_text(_record("m1", 500) + "\n" + _record("m2", 0) + "\n", encoding="utf-8")
        self.db_path = self.tmp / "usage.db"

    def _one_hour(self):
        conn = get_db(self.db_path)
        rows = dict(conn.execute("SELECT message_id, cache_creation_1h_tokens FROM turns"))
        conn.close()
        return rows

    def test_parse_records_the_split(self):
        _, turns, _, _ = parse_jsonl_file(str(self.transcript))
        self.assertEqual({t["message_id"]: t["cache_creation_1h_tokens"] for t in turns},
                         {"m1": 500, "m2": 0})

    def test_scan_stores_the_split(self):
        scan(projects_dir=self.projects, db_path=self.db_path, verbose=False)
        self.assertEqual(self._one_hour(), {"m1": 500, "m2": 0})

    def test_backfill_fills_turns_scanned_before_the_column(self):
        scan(projects_dir=self.projects, db_path=self.db_path, verbose=False)
        conn = get_db(self.db_path)
        conn.execute("UPDATE turns SET cache_creation_1h_tokens = 0")
        conn.execute("DELETE FROM schema_meta WHERE key = 'cache_1h_backfill_done'")
        conn.commit()
        conn.close()
        # The file is unchanged, so only the backfill can restore the value.
        scan(projects_dir=self.projects, db_path=self.db_path, verbose=False)
        self.assertEqual(self._one_hour(), {"m1": 500, "m2": 0})


class TestSessionByModel(unittest.TestCase):
    def setUp(self):
        self.data = dashboard.get_dashboard_data(_mixed_db())
        self.session = self.data["sessions_all"][0]

    def test_rows_split_by_model_and_thread(self):
        keys = {(r["model"], r["is_subagent"]) for r in self.session["by_model"]}
        self.assertEqual(keys, {("claude-fable-5-2", 0), ("claude-opus-5-5", 1), ("claude-haiku-5-5", 1)})

    def test_rows_add_up_to_the_session_totals(self):
        for key in ("input", "output", "cache_read", "cache_creation", "turns"):
            self.assertEqual(sum(r[key] for r in self.session["by_model"]), self.session[key], key)

    def test_rows_carry_one_hour_cache_writes(self):
        main = next(r for r in self.session["by_model"] if not r["is_subagent"])
        self.assertEqual((main["cache_creation"], main["cache_creation_1h"]), (405_000, 400_000))
        for key in ("daily_by_model", "subagent_by_type", "top_dispatches"):
            self.assertIn("cache_creation_1h", self.data[key][0], key)

    def test_only_the_long_prompt_turn_is_in_the_long_sums(self):
        expected = {"input": 20, "output": 7_000, "cache_read": 275_000,
                    "cache_creation": 50_000, "cache_creation_1h": 30_000}
        haiku = next(r for r in self.session["by_model"] if r["model"] == "claude-haiku-5-5")
        self.assertEqual(haiku["long"], expected)
        self.assertEqual(haiku["input"], 10_020)  # the totals still hold both turns
        dispatch = next(d for d in self.data["top_dispatches"] if d["agent_id"] == "agent-2")
        self.assertEqual(dispatch["long"], expected)
        for row in self.session["by_model"]:
            if row["model"] != "claude-haiku-5-5":
                self.assertNotIn("long", row, row["model"])


class TestLongPromptTier(unittest.TestCase):
    def test_long_prompt_turn_costs_five_times(self):
        base = calc_cost("claude-haiku-5-5", 20, 7_000, 275_000, 50_000, 30_000)
        self.assertAlmostEqual(_turn_cost(LONG_TURN), 5 * base)
        # Input $0.50 and output $2.50 per MTok on the long tier.
        row = {"model": "claude-haiku-5-5", "inp": 1_000_000, "out": 1_000_000, "cr": 0, "cc": 0, "cc1h": 0,
               "long_input": 1_000_000, "long_output": 1_000_000, "long_cache_read": 0,
               "long_cache_creation": 0, "long_cache_creation_1h": 0}
        self.assertAlmostEqual(cli.row_cost(row), 3.00)

    def test_cli_stats_total_matches_the_per_turn_sum(self):
        out = io.StringIO()
        with mock.patch.object(cli, "DB_PATH", _mixed_db()), contextlib.redirect_stdout(out):
            cli.cmd_stats()
        self.assertIn(f"Est. total cost:  ${_expected_cost(MIXED_TURNS):.4f}", out.getvalue())


# Runs the dashboard's own JS cost functions in Node on the same data.
JS_HARNESS = """
const data = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const s = data.sessions_all[0];
const sum = rows => rows.reduce((a, r) => a + rowCost(r), 0);
console.log(JSON.stringify({
  haiku: sum(s.by_model.filter(r => r.model === 'claude-haiku-5-5')),
  session: sessionCost(s),
  main: sum(s.by_model.filter(r => !r.is_subagent)),
  dispatches: sum(data.top_dispatches),
  daily: sum(data.daily_by_model),
  dated_fable_5: getPricing('claude-fable-5-20260601').input,
  fable_5_2: getPricing('claude-fable-5-2-20261101').input,
}));
"""


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class TestDashboardJsCost(unittest.TestCase):
    """The session row, the session's parts and the daily totals must agree
    with each other and with the CLI."""

    @classmethod
    def setUpClass(cls):
        html = dashboard.HTML_TEMPLATE
        start = html.index("const PRICING = {")
        end = html.index("// ── Formatting", start)
        data = dashboard.get_dashboard_data(_mixed_db())
        result = subprocess.run(["node", "-e", html[start:end] + JS_HARNESS],
                                input=json.dumps(data), capture_output=True,
                                text=True, encoding="utf-8", timeout=60)
        if result.returncode != 0:
            raise AssertionError(result.stderr)
        cls.js = json.loads(result.stdout)

    def test_session_cost_is_the_sum_of_its_models(self):
        self.assertAlmostEqual(self.js["session"], _expected_cost(MIXED_TURNS), places=6)

    def test_session_cost_is_not_the_primary_model_rate(self):
        at_fable_rate = calc_cost("claude-fable-5-2", _total("input_tokens"), _total("output_tokens"),
                                  _total("cache_read_tokens"), _total("cache_creation_tokens"),
                                  _total("cache_creation_1h_tokens"))
        self.assertLess(self.js["session"], at_fable_rate * 0.8)

    def test_main_plus_subagents_equals_session(self):
        self.assertAlmostEqual(self.js["main"] + self.js["dispatches"], self.js["session"], places=6)
        self.assertAlmostEqual(self.js["main"], _expected_cost(MIXED_TURNS[:2]), places=6)

    def test_daily_total_equals_session(self):
        self.assertAlmostEqual(self.js["daily"], self.js["session"], places=6)

    def test_long_prompt_tier_matches_cli(self):
        self.assertAlmostEqual(self.js["haiku"], _expected_cost(MIXED_TURNS[3:]), places=6)

    def test_js_price_lookup_matches_cli(self):
        self.assertEqual(self.js["dated_fable_5"], 10.00)
        self.assertEqual(self.js["fable_5_2"], 8.00)


if __name__ == "__main__":
    unittest.main()
