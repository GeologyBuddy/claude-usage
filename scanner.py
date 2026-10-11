"""
scanner.py - Scans Claude Code JSONL transcript files and stores data in SQLite.
"""

import json
import os
import sys
import glob
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import unquote, urlparse

# Single source of truth for the app version reported by the CLI (`--version`)
# and the dashboard footer. CHANGELOG.md is the canonical version reference, but
# it isn't bundled into the .vsix — only the three Python files are — so the
# runtime version has to live here as a constant. Keep this in lockstep with the
# top CHANGELOG heading and vscode-extension/package.json (a parity test guards
# all three; see tests/test_version.py).
VERSION = "1.6.0"

PROJECTS_DIR = Path.home() / ".claude" / "projects"
XCODE_PROJECTS_DIR = Path.home() / "Library" / "Developer" / "Xcode" / "CodingAssistant" / "ClaudeAgentConfig" / "projects"
DB_PATH = Path(os.environ.get("CLAUDE_USAGE_DB", Path.home() / ".claude" / "usage.db"))
DEFAULT_PROJECTS_DIRS = [PROJECTS_DIR, XCODE_PROJECTS_DIR]

# Higher number = higher priority when choosing a session's primary model.
# Fable / Mythos are Anthropic's most capable class, so they outrank Opus.
MODEL_PRIORITY = {"fable": 5, "mythos": 5, "opus": 3, "sonnet": 2, "haiku": 1}


def _model_priority(model):
    """Return a priority score for a model name (higher = more capable)."""
    if not model:
        return 0
    m = model.lower()
    for keyword, priority in MODEL_PRIORITY.items():
        if keyword in m:
            return priority
    return 0


def get_db(db_path=DB_PATH):
    # Ensure the parent directory exists — on a fresh install or CI runner
    # ~/.claude may not yet exist, and sqlite3.connect needs the parent dir.
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id      TEXT PRIMARY KEY,
            project_name    TEXT,
            first_timestamp TEXT,
            last_timestamp  TEXT,
            git_branch      TEXT,
            total_input_tokens      INTEGER DEFAULT 0,
            total_output_tokens     INTEGER DEFAULT 0,
            total_cache_read        INTEGER DEFAULT 0,
            total_cache_creation    INTEGER DEFAULT 0,
            model           TEXT,
            turn_count      INTEGER DEFAULT 0,
            topic           TEXT
        );

        CREATE TABLE IF NOT EXISTS turns (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id              TEXT,
            timestamp               TEXT,
            model                   TEXT,
            input_tokens            INTEGER DEFAULT 0,
            output_tokens           INTEGER DEFAULT 0,
            cache_read_tokens       INTEGER DEFAULT 0,
            cache_creation_tokens   INTEGER DEFAULT 0,
            tool_name               TEXT,
            cwd                     TEXT,
            message_id              TEXT,
            is_subagent             INTEGER DEFAULT 0,
            agent_id                TEXT
        );

        CREATE TABLE IF NOT EXISTS processed_files (
            path    TEXT PRIMARY KEY,
            mtime   REAL,
            lines   INTEGER
        );

        CREATE TABLE IF NOT EXISTS agents (
            agent_id              TEXT PRIMARY KEY,
            agent_type            TEXT,
            dispatched_in_session TEXT,
            completed_at          TEXT,
            status                TEXT,
            total_tokens          INTEGER,
            total_duration_ms     INTEGER,
            tool_use_count        INTEGER
        );

        CREATE TABLE IF NOT EXISTS schema_meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS skill_events (
            uuid        TEXT PRIMARY KEY,
            session_id  TEXT,
            timestamp   TEXT,
            kind        TEXT,
            skill       TEXT,
            source      TEXT,
            load_chars  INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_skill_events_session
            ON skill_events(session_id, timestamp);

        -- One row per GitHub Copilot Chat request (see scan_copilot).
        CREATE TABLE IF NOT EXISTS copilot_requests (
            request_id    TEXT PRIMARY KEY,
            session_id    TEXT,
            workspace     TEXT,
            title         TEXT,
            timestamp     TEXT,
            model         TEXT,
            prompt_tokens INTEGER,
            output_tokens INTEGER,
            credits       REAL,
            elapsed_ms    INTEGER,
            rounds        INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_copilot_session ON copilot_requests(session_id);

        CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
        CREATE INDEX IF NOT EXISTS idx_turns_timestamp ON turns(timestamp);
        CREATE INDEX IF NOT EXISTS idx_sessions_first ON sessions(first_timestamp);
        CREATE INDEX IF NOT EXISTS idx_agents_type ON agents(agent_type);
    """)
    # Add message_id column if upgrading from older schema
    try:
        conn.execute("SELECT message_id FROM turns LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE turns ADD COLUMN message_id TEXT")
    # Subagent attribution columns (added in a later schema version)
    _ensure_column(conn, "turns", "is_subagent", "INTEGER DEFAULT 0")
    _ensure_column(conn, "turns", "agent_id", "TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_subagent ON turns(is_subagent)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_agent_id ON turns(agent_id)")
    # Session topic (from custom-title / ai-title records; added in a later
    # schema version). The one-time backfill of pre-existing sessions is driven
    # by scan() via the schema_meta 'topic_backfill_done' marker (not by the
    # column-add event), so it also covers DBs that gained the column from an
    # earlier build that predated the backfill.
    _ensure_column(conn, "sessions", "topic", "TEXT")
    # Conditional unique index: only dedup non-null message IDs
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_turns_message_id
        ON turns(message_id) WHERE message_id IS NOT NULL AND message_id != ''
    """)
    conn.commit()


def _ensure_column(conn, table, column, decl):
    """Add a column to an existing table if it isn't already present.

    Returns True if the column was just added (an upgrade of an existing DB),
    False if it was already there (fresh DB or already-migrated).
    """
    cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        return True
    return False


def _meta_get(conn, key):
    """Read a value from the schema_meta key/value table (None if absent)."""
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def _meta_set(conn, key, value):
    """Upsert a value into the schema_meta key/value table."""
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
        (key, value))


def _extract_title(record):
    """Extract a session title from a custom-title or ai-title record."""
    rtype = record.get("type")
    if rtype == "custom-title":
        return record.get("customTitle")
    if rtype == "ai-title":
        return record.get("aiTitle")
    return None


def _backfill_topics(conn, jsonl_files):
    """One-time backfill of topics for a DB created before topic support.

    Transcript files scanned before the topic column existed are already in
    processed_files, so an incremental scan skips them and never sees the
    custom-title / ai-title records they already contain. Re-read just those
    records (turns are left untouched, so token totals cannot drift) and set the
    topic for any session that doesn't have one yet. Runs once, gated by a flag
    in schema_meta (see scan()). Returns the number of sessions filled.
    """
    needing = {r["session_id"] for r in conn.execute(
        "SELECT session_id FROM sessions WHERE topic IS NULL OR topic = ''")}
    if not needing:
        return 0

    titles = {}          # session_id -> chosen title
    has_custom = set()   # sessions whose topic came from a custom-title record
    for filepath in jsonl_files:
        try:
            with open(filepath, encoding="utf-8", errors="replace") as f:
                for line in f:
                    # Cheap prefilter: only title records carry the substring
                    # "title" (in their "custom-title" / "ai-title" type), so we
                    # skip JSON-parsing the ~99% of lines that are turns.
                    if "title" not in line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    title = _extract_title(record)
                    if not title:
                        continue
                    sid = record.get("sessionId")
                    if sid not in needing:
                        continue
                    # custom-title wins; ai-title only if no custom-title seen.
                    if record.get("type") == "custom-title":
                        titles[sid] = title
                        has_custom.add(sid)
                    elif sid not in has_custom:
                        titles.setdefault(sid, title)
        except Exception as e:
            print(f"  Warning: error reading {filepath}: {e}")

    for sid, title in titles.items():
        conn.execute(
            "UPDATE sessions SET topic = ? WHERE session_id = ? "
            "AND (topic IS NULL OR topic = '')", (title, sid))
    conn.commit()
    return len(titles)


SKILL_BODY_PREFIX = "Base directory for this skill:"


def _is_project_skill_dir(base_dir):
    """True unless the skill lives under ~/.claude or ~/.agents (user-level
    skills and plugin skills, which Claude Code caches in ~/.claude/plugins).
    base_dir must already be normalized to forward slashes."""
    norm = base_dir.lower()
    home = str(Path.home()).replace("\\", "/").rstrip("/").lower()
    return not any(norm.startswith(f"{home}/{d}/") for d in (".claude", ".agents"))


def _user_text(record):
    """Return the text of a user record's message, or None if it carries a
    tool_result (a tool reply, not something the user typed)."""
    content = (record.get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    texts = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "tool_result":
            return None
        if item.get("type") == "text":
            texts.append(item.get("text") or "")
    return "\n".join(texts) if texts else None


def extract_skill_event(record):
    """Classify a main-thread user record as a skill load or a real prompt.

    Both the Skill tool and a typed /skill-name inject the SKILL.md body as an
    isMeta user record starting "Base directory for this skill: <path>"; the
    tool path also sets sourceToolUseID. Only project skills are kept. A real
    prompt (non-meta user text) is recorded so the dashboard knows where a
    skill's active window ends — its text is not stored.
    """
    if record.get("type") != "user" or record.get("isSidechain"):
        return None
    uuid, session_id = record.get("uuid"), record.get("sessionId")
    if not uuid or not session_id:
        return None
    text = _user_text(record)
    if text is None:
        return None
    event = {"uuid": uuid, "session_id": session_id,
             "timestamp": record.get("timestamp", ""),
             "kind": "prompt", "skill": None, "source": None, "load_chars": None}
    if record.get("isMeta"):
        if not text.startswith(SKILL_BODY_PREFIX):
            return None
        first_line = text[len(SKILL_BODY_PREFIX):].split("\n", 1)[0]
        base_dir = first_line.strip().replace("\\", "/").rstrip("/")
        if not _is_project_skill_dir(base_dir):
            return None
        event.update(kind="skill",
                     skill=base_dir.rsplit("/", 1)[-1],
                     source="auto" if record.get("sourceToolUseID") else "user",
                     load_chars=len(text))
    return event


def parse_skill_events(filepath, skip_lines=0):
    """Read skill-load and prompt events from a transcript, past skip_lines."""
    events = []
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            for n, line in enumerate(f, 1):
                # Cheap prefilter: every event is a user record.
                if n <= skip_lines or '"user"' not in line:
                    continue
                try:
                    event = extract_skill_event(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if event:
                    events.append(event)
    except Exception as e:
        print(f"  Warning: error reading {filepath}: {e}")
    return events


def insert_skill_events(conn, events):
    conn.executemany("""
        INSERT OR IGNORE INTO skill_events
            (uuid, session_id, timestamp, kind, skill, source, load_chars)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [(e["uuid"], e["session_id"], e["timestamp"], e["kind"],
           e["skill"], e["source"], e["load_chars"]) for e in events])


def _backfill_skill_events(conn):
    """One-time skill-event backfill for transcripts already in processed_files
    (an incremental scan would never revisit them). Gated in scan()."""
    paths = [r["path"] for r in conn.execute("SELECT path FROM processed_files")]
    for path in paths:
        if os.path.exists(path):
            insert_skill_events(conn, parse_skill_events(path))
    conn.commit()


def project_name_from_cwd(cwd):
    """Derive a friendly project name from cwd path."""
    if not cwd:
        return "unknown"
    # Normalize to forward slashes, take last 2 components
    parts = cwd.replace("\\", "/").rstrip("/").split("/")
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return parts[-1] if parts else "unknown"


def is_subagent_record(record, source_path=""):
    """True if a record belongs to a dispatched subagent (Task/Agent tool).

    Subagents are detected three ways: an explicit ``isSidechain`` flag, an
    ``agentId`` on the record (or its ``data`` wrapper), or a transcript path
    under a ``subagents`` directory (Claude Code writes one jsonl per subagent).
    """
    if record.get("isSidechain"):
        return True
    if record.get("agentId"):
        return True
    data = record.get("data")
    if isinstance(data, dict) and data.get("agentId"):
        return True
    sp = str(source_path).replace("\\", "/").lower()
    return "/subagents/" in sp


def record_agent_id(record):
    """Pull the subagent id off a record, if any (top-level or data wrapper)."""
    agent_id = record.get("agentId")
    if not agent_id:
        data = record.get("data")
        if isinstance(data, dict):
            agent_id = data.get("agentId")
    return agent_id


def extract_agent_dispatch(record):
    """Pull subagent identity from a parent's tool_result record.

    Claude Code writes a ``toolUseResult`` dict on the user-side record that
    closes out an Agent/Task tool invocation. It carries ``agentId`` (matching
    the subagent jsonl's records) and ``agentType`` (the human-readable type
    such as 'general-purpose' or 'Explore') plus aggregate stats.
    """
    if record.get("type") != "user":
        return None
    tur = record.get("toolUseResult")
    if not isinstance(tur, dict):
        return None
    agent_id = tur.get("agentId")
    agent_type = tur.get("agentType")
    is_async = bool(tur.get("isAsync"))
    # A background (async) launch has no agentType and no stats yet; its type
    # comes from the subagent's .meta.json (see _agents_from_meta).
    if not agent_id or not (agent_type or is_async):
        return None
    return {
        "agent_id": agent_id,
        "agent_type": agent_type,
        "dispatched_in_session": record.get("sessionId"),
        "completed_at": record.get("timestamp", ""),
        "status": None if is_async else tur.get("status"),
        "total_tokens": tur.get("totalTokens"),
        "total_duration_ms": tur.get("totalDurationMs"),
        "tool_use_count": tur.get("totalToolUseCount"),
    }


def _agents_from_meta(jsonl_files):
    """Agent types from subagents/agent-<id>.meta.json files.

    Background agents launch with no agentType on the parent record; Claude
    Code writes it to the meta file beside the subagent transcript instead.
    Read on every scan (the files are tiny), so already-scanned agents get
    their type too.
    """
    agents = []
    for p in jsonl_files:
        path = Path(p)
        if path.parent.name != "subagents" or not path.stem.startswith("agent-"):
            continue
        try:
            meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if meta.get("agentType"):
            agents.append({"agent_id": path.stem[len("agent-"):],
                           "agent_type": meta["agentType"],
                           "dispatched_in_session": path.parent.parent.name})
    return agents


def _fill_unknown_projects(conn):
    """Name sessions stuck at project 'unknown' from their turns' cwd.

    A session's project comes from its first record, and when that is an
    ai-title / custom-title record (no cwd) it stays 'unknown'. Turns keep the
    cwd, so repair from them; cheap, because only 'unknown' rows are touched.
    """
    rows = conn.execute("""
        SELECT s.session_id,
               (SELECT t.cwd FROM turns t WHERE t.session_id = s.session_id
                  AND t.cwd != '' ORDER BY t.timestamp LIMIT 1) AS cwd
        FROM sessions s WHERE s.project_name = 'unknown'
    """).fetchall()
    conn.executemany("UPDATE sessions SET project_name = ? WHERE session_id = ?",
                     [(project_name_from_cwd(r["cwd"]), r["session_id"]) for r in rows if r["cwd"]])


def upsert_agents(conn, agents):
    """Insert or update agent dispatch metadata. Last non-null write wins per
    field, so a sparse record (async launch, meta file) never erases a value."""
    if not agents:
        return
    conn.executemany("""
        INSERT INTO agents
            (agent_id, agent_type, dispatched_in_session, completed_at,
             status, total_tokens, total_duration_ms, tool_use_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(agent_id) DO UPDATE SET
            agent_type            = COALESCE(excluded.agent_type,            agents.agent_type),
            dispatched_in_session = COALESCE(excluded.dispatched_in_session, agents.dispatched_in_session),
            completed_at          = COALESCE(excluded.completed_at,          agents.completed_at),
            status                = COALESCE(excluded.status,                agents.status),
            total_tokens          = COALESCE(excluded.total_tokens,          agents.total_tokens),
            total_duration_ms     = COALESCE(excluded.total_duration_ms,     agents.total_duration_ms),
            tool_use_count        = COALESCE(excluded.tool_use_count,        agents.tool_use_count)
    """, [
        (a["agent_id"], a["agent_type"], a.get("dispatched_in_session"),
         a.get("completed_at"), a.get("status"),
         a.get("total_tokens"), a.get("total_duration_ms"), a.get("tool_use_count"))
        for a in agents
    ])


def parse_jsonl_file(filepath):
    """Parse a JSONL file and return (session_metas, turns, agents, line_count).

    Deduplicates streaming events by message.id — Claude Code logs multiple
    JSONL records per API response, all sharing the same message.id. Only the
    last record per message_id is kept (it has the final usage tallies).
    """
    seen_messages = {}  # message_id -> turn dict (dedup streaming records)
    turns_no_id = []    # turns without a message_id (kept as-is)
    session_meta = {}   # session_id -> dict
    agents = {}         # agent_id -> dispatch dict
    line_count = 0

    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            for line_count, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                rtype = record.get("type")
                if rtype not in ("assistant", "user", "custom-title", "ai-title"):
                    continue

                session_id = record.get("sessionId")
                if not session_id:
                    continue

                # Extract session title from title records
                title = _extract_title(record)
                if title:
                    if session_id not in session_meta:
                        session_meta[session_id] = {
                            "session_id": session_id,
                            "project_name": "unknown",
                            "first_timestamp": "",
                            "last_timestamp": "",
                            "git_branch": "",
                            "model": None,
                            "topic": None,
                        }
                    meta = session_meta[session_id]
                    # custom-title always wins; ai-title only if no custom-title set
                    if rtype == "custom-title":
                        meta["topic"] = title
                    elif rtype == "ai-title" and not meta.get("topic"):
                        meta["topic"] = title
                    continue

                if rtype == "user":
                    dispatch = extract_agent_dispatch(record)
                    if dispatch is not None:
                        agents[dispatch["agent_id"]] = dispatch

                timestamp = record.get("timestamp", "")
                cwd = record.get("cwd", "")
                git_branch = record.get("gitBranch", "")

                # Update session metadata from any record
                if session_id not in session_meta:
                    session_meta[session_id] = {
                        "session_id": session_id,
                        "project_name": project_name_from_cwd(cwd),
                        "first_timestamp": timestamp,
                        "last_timestamp": timestamp,
                        "git_branch": git_branch,
                        "model": None,
                        "topic": None,
                    }
                else:
                    meta = session_meta[session_id]
                    if timestamp and (not meta["first_timestamp"] or timestamp < meta["first_timestamp"]):
                        meta["first_timestamp"] = timestamp
                    if timestamp and (not meta["last_timestamp"] or timestamp > meta["last_timestamp"]):
                        meta["last_timestamp"] = timestamp
                    if git_branch and not meta["git_branch"]:
                        meta["git_branch"] = git_branch

                if rtype == "assistant":
                    msg = record.get("message", {})
                    usage = msg.get("usage", {})
                    model = msg.get("model", "")
                    message_id = msg.get("id", "")

                    input_tokens = usage.get("input_tokens", 0) or 0
                    output_tokens = usage.get("output_tokens", 0) or 0
                    cache_read = usage.get("cache_read_input_tokens", 0) or 0
                    cache_creation = usage.get("cache_creation_input_tokens", 0) or 0

                    # Only record turns that have actual token usage
                    if input_tokens + output_tokens + cache_read + cache_creation == 0:
                        continue

                    # Extract tool name from content if present
                    tool_name = None
                    for item in msg.get("content", []):
                        if isinstance(item, dict) and item.get("type") == "tool_use":
                            tool_name = item.get("name")
                            break

                    if model:
                        session_meta[session_id]["model"] = model

                    turn = {
                        "session_id": session_id,
                        "timestamp": timestamp,
                        "model": model,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cache_read_tokens": cache_read,
                        "cache_creation_tokens": cache_creation,
                        "tool_name": tool_name,
                        "cwd": cwd,
                        "message_id": message_id,
                        "is_subagent": 1 if is_subagent_record(record, filepath) else 0,
                        "agent_id": record_agent_id(record),
                    }

                    # Dedup: last record per message_id wins (final usage tallies)
                    if message_id:
                        seen_messages[message_id] = turn
                    else:
                        turns_no_id.append(turn)

    except Exception as e:
        print(f"  Warning: error reading {filepath}: {e}")

    turns = turns_no_id + list(seen_messages.values())
    return list(session_meta.values()), turns, list(agents.values()), line_count


def aggregate_sessions(session_metas, turns):
    """Aggregate turn data back into session-level stats."""
    from collections import defaultdict, Counter

    session_stats = defaultdict(lambda: {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cache_read": 0,
        "total_cache_creation": 0,
        "turn_count": 0,
        "model": None,
    })
    session_model_counts = defaultdict(Counter)

    for t in turns:
        s = session_stats[t["session_id"]]
        s["total_input_tokens"] += t["input_tokens"]
        s["total_output_tokens"] += t["output_tokens"]
        s["total_cache_read"] += t["cache_read_tokens"]
        s["total_cache_creation"] += t["cache_creation_tokens"]
        s["turn_count"] += 1
        if t["model"]:
            session_model_counts[t["session_id"]][t["model"]] += 1

    for sid, counts in session_model_counts.items():
        if counts:
            session_stats[sid]["model"] = counts.most_common(1)[0][0]

    # Merge into session_metas
    result = []
    for meta in session_metas:
        sid = meta["session_id"]
        stats = session_stats[sid]
        result.append({**meta, **stats})
    return result


def upsert_sessions(conn, sessions):
    for s in sessions:
        # Check if session exists
        existing = conn.execute(
            "SELECT total_input_tokens, total_output_tokens, total_cache_read, "
            "total_cache_creation, turn_count FROM sessions WHERE session_id = ?",
            (s["session_id"],)
        ).fetchone()

        # A session seen only via a title record (custom-title / ai-title carry a
        # sessionId but no timestamp) has no real content. Don't let it INSERT a
        # phantom, token-less row; if the session already exists it still falls
        # through to the UPDATE below and sets its topic.
        if existing is None and not s.get("first_timestamp"):
            continue

        if existing is None:
            conn.execute("""
                INSERT INTO sessions
                    (session_id, project_name, first_timestamp, last_timestamp,
                     git_branch, total_input_tokens, total_output_tokens,
                     total_cache_read, total_cache_creation, model, turn_count,
                     topic)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                s["session_id"], s["project_name"], s["first_timestamp"],
                s["last_timestamp"], s["git_branch"],
                s["total_input_tokens"], s["total_output_tokens"],
                s["total_cache_read"], s["total_cache_creation"],
                s["model"], s["turn_count"], s.get("topic")
            ))
        else:
            # Update: add new tokens on top of existing (since we only insert new turns)
            # Keep the highest-priority model (e.g. opus over haiku from subagents)
            existing_row = conn.execute(
                "SELECT model, topic FROM sessions WHERE session_id = ?",
                (s["session_id"],)
            ).fetchone()
            existing_model = existing_row["model"]
            new_model = s["model"]
            if _model_priority(new_model) > _model_priority(existing_model):
                model_to_set = new_model
            else:
                model_to_set = existing_model

            # Update topic if the new scan found one and the existing is empty
            new_topic = s.get("topic")
            existing_topic = existing_row["topic"]
            topic_to_set = new_topic if new_topic else existing_topic

            conn.execute("""
                UPDATE sessions SET
                    last_timestamp = MAX(last_timestamp, ?),
                    total_input_tokens = total_input_tokens + ?,
                    total_output_tokens = total_output_tokens + ?,
                    total_cache_read = total_cache_read + ?,
                    total_cache_creation = total_cache_creation + ?,
                    turn_count = turn_count + ?,
                    model = ?,
                    topic = ?
                WHERE session_id = ?
            """, (
                s["last_timestamp"],
                s["total_input_tokens"], s["total_output_tokens"],
                s["total_cache_read"], s["total_cache_creation"],
                s["turn_count"], model_to_set, topic_to_set,
                s["session_id"]
            ))


def insert_turns(conn, turns):
    conn.executemany("""
        INSERT OR IGNORE INTO turns
            (session_id, timestamp, model, input_tokens, output_tokens,
             cache_read_tokens, cache_creation_tokens, tool_name, cwd, message_id,
             is_subagent, agent_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (t["session_id"], t["timestamp"], t["model"],
         t["input_tokens"], t["output_tokens"],
         t["cache_read_tokens"], t["cache_creation_tokens"],
         t["tool_name"], t["cwd"], t.get("message_id", ""),
         t.get("is_subagent", 0), t.get("agent_id"))
        for t in turns
    ])


# ── GitHub Copilot Chat (VS Code) ───────────────────────────────────────────
# VS Code saves each Copilot chat as a JSONL change log in
# <User>/workspaceStorage/<hash>/chatSessions/ (or globalStorage/
# emptyWindowChatSessions/ when no folder is open). Line 1 (kind 0) is a full
# snapshot; later lines patch it: kind 1 sets the value at path k, kind 2
# appends v to the array at k (after cutting it to index i, when given).
# Token meaning, checked against real logs: completionTokens is the total over
# all tool-call rounds of a request; promptTokens is one call's prompt (the
# context size), not a sum. copilotCredits is the unit Copilot bills.

def _vscode_user_dirs():
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return [base / name / "User" for name in ("Code", "Code - Insiders", "VSCodium")]


# Read at call time (not frozen as a default arg) so tests can patch it.
COPILOT_DIRS = _vscode_user_dirs()


def replay_copilot_chat(filepath):
    """Apply a chat's change log and return the final chat state (a dict)."""
    state = {}
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            kind, path, value = rec.get("kind"), rec.get("k") or [], rec.get("v")
            if kind == 0:
                state = value
                continue
            target = state
            for key in path[:-1]:
                target = target[key]
            if kind == 1:
                target[path[-1]] = value
            elif kind == 2:
                arr = target[path[-1]]
                if "i" in rec:
                    del arr[rec["i"]:]
                arr.extend(value)
    return state


def _copilot_workspace(chat_file):
    """Friendly name of the folder a chat belongs to, from workspace.json."""
    ws_json = Path(chat_file).parent.parent / "workspace.json"
    try:
        info = json.loads(ws_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "(no folder)"
    uri = info.get("folder") or info.get("workspace") or ""
    path = unquote(urlparse(uri).path)
    if len(path) > 2 and path[0] == "/" and path[2] == ":":  # /c:/Users/... on Windows
        path = path[1:]
    return project_name_from_cwd(path)


def _ms_to_iso(ms):
    """Epoch ms -> the same ISO-8601 UTC form Claude Code uses (…T..:..:..mmmZ)."""
    dt = datetime.fromtimestamp(ms / 1000, timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(ms) % 1000:03d}Z"


def parse_copilot_chat(filepath):
    """Return (session_id, request rows) for one Copilot chat file."""
    state = replay_copilot_chat(filepath)
    session_id = state.get("sessionId") or Path(filepath).stem
    requests = state.get("requests") or []
    questions = [((q.get("message") or {}).get("text") or "").strip() for q in requests]
    title = state.get("customTitle") or next((t for t in questions if t), "")[:80]
    workspace = _copilot_workspace(filepath)
    rows = []
    for q in requests:
        prompt, output, credits = q.get("promptTokens"), q.get("completionTokens"), q.get("copilotCredits")
        if (prompt is None and output is None and credits is None) or not q.get("timestamp"):
            continue  # cancelled before the model answered
        meta = (q.get("result") or {}).get("metadata") or {}
        model = meta.get("resolvedModel") or q.get("modelId") or "unknown"
        if model.startswith("copilot/"):
            model = model[len("copilot/"):]
        rows.append({
            "request_id": q.get("requestId") or f"{session_id}:{q['timestamp']}",
            "session_id": session_id, "workspace": workspace, "title": title,
            "timestamp": _ms_to_iso(q["timestamp"]), "model": model,
            "prompt_tokens": prompt or 0, "output_tokens": output or 0,
            "credits": credits or 0.0, "elapsed_ms": q.get("elapsedMs") or 0,
            "rounds": len(meta.get("toolCallRounds") or []),
        })
    return session_id, rows


def _copilot_chat_files(user_dirs):
    files = []
    for d in map(Path, user_dirs):
        files += glob.glob(str(d / "workspaceStorage" / "*" / "chatSessions" / "*.jsonl"))
        files += glob.glob(str(d / "globalStorage" / "emptyWindowChatSessions" / "*.jsonl"))
    return sorted(files)


def scan_copilot(conn, user_dirs, verbose=False):
    """Ingest new or changed Copilot chat files. Returns the number of files read.

    A changed chat is re-read whole (the files are small) and its rows are
    replaced, so edits and undone requests are reflected. Rows of chats whose
    file is gone are kept, like Claude history after transcript pruning.
    """
    read = 0
    for filepath in _copilot_chat_files(user_dirs):
        try:
            mtime = os.path.getmtime(filepath)
        except OSError:
            continue
        row = conn.execute("SELECT mtime FROM processed_files WHERE path = ?", (filepath,)).fetchone()
        if row and abs(row["mtime"] - mtime) < 0.01:
            continue
        try:
            session_id, rows = parse_copilot_chat(filepath)
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
            if verbose:
                print(f"  Warning: skipped Copilot chat {filepath}: {e}")
            continue
        conn.execute("DELETE FROM copilot_requests WHERE session_id = ?", (session_id,))
        conn.executemany("""
            INSERT OR REPLACE INTO copilot_requests
                (request_id, session_id, workspace, title, timestamp, model,
                 prompt_tokens, output_tokens, credits, elapsed_ms, rounds)
            VALUES (:request_id, :session_id, :workspace, :title, :timestamp, :model,
                    :prompt_tokens, :output_tokens, :credits, :elapsed_ms, :rounds)
        """, rows)
        conn.execute("INSERT OR REPLACE INTO processed_files (path, mtime, lines) VALUES (?, ?, 0)",
                     (filepath, mtime))
        conn.commit()
        read += 1
        if verbose:
            print(f"  [COPILOT] {filepath} ({len(rows)} requests)")
    return read


def scan(projects_dir=None, projects_dirs=None, db_path=DB_PATH, verbose=True, copilot_dirs=None):
    conn = get_db(db_path)
    init_db(conn)

    if projects_dirs:
        dirs_to_scan = [Path(d) for d in projects_dirs]
    elif projects_dir:
        dirs_to_scan = [Path(projects_dir)]
    else:
        dirs_to_scan = DEFAULT_PROJECTS_DIRS

    jsonl_files = []
    for d in dirs_to_scan:
        if not d.exists():
            continue
        if verbose:
            print(f"Scanning {d} ...")
        jsonl_files.extend(glob.glob(str(d / "**" / "*.jsonl"), recursive=True))
    jsonl_files.sort()

    # One-time topic backfill for DBs whose sessions predate topic support: fill
    # topics from title records in already-processed transcripts that an
    # incremental scan would otherwise never revisit. Runs once, gated by the
    # schema_meta 'topic_backfill_done' marker. It runs before the main loop, so
    # on a fresh DB the sessions table is still empty and this no-ops; only DBs
    # with pre-existing untitled sessions do real work.
    if _meta_get(conn, "topic_backfill_done") != "1":
        filled = _backfill_topics(conn, jsonl_files)
        _meta_set(conn, "topic_backfill_done", "1")
        conn.commit()
        if verbose and filled:
            print(f"Backfilled topic for {filled} existing session(s).")

    # Same idea for project-skill events (added after topics): fill them in for
    # transcripts that were processed before the skill_events table existed.
    if _meta_get(conn, "skill_backfill_done") != "1":
        _backfill_skill_events(conn)
        _meta_set(conn, "skill_backfill_done", "1")
        conn.commit()

    new_files = 0
    updated_files = 0
    skipped_files = 0
    total_turns = 0
    total_sessions = set()

    for filepath in jsonl_files:
        try:
            mtime = os.path.getmtime(filepath)
        except OSError:
            continue

        row = conn.execute(
            "SELECT mtime, lines FROM processed_files WHERE path = ?",
            (filepath,)
        ).fetchone()

        if row and abs(row["mtime"] - mtime) < 0.01:
            skipped_files += 1
            continue

        is_new = row is None
        if verbose:
            status = "NEW" if is_new else "UPD"
            print(f"  [{status}] {filepath}")

        if is_new:
            # New file: full parse (single read, returns line count)
            session_metas, turns, agents, line_count = parse_jsonl_file(filepath)
            upsert_agents(conn, agents)
            insert_skill_events(conn, parse_skill_events(filepath))

            if turns or session_metas:
                sessions = aggregate_sessions(session_metas, turns)
                upsert_sessions(conn, sessions)
                insert_turns(conn, turns)
                for s in sessions:
                    total_sessions.add(s["session_id"])
                total_turns += len(turns)
                new_files += 1

        else:
            # Updated file: read once, process only new lines
            old_lines = row["lines"] if row else 0
            seen_messages = {}  # message_id -> turn (dedup streaming)
            turns_no_id = []
            new_session_metas = {}
            agents = {}         # agent_id -> dispatch dict
            line_count = 0

            try:
                with open(filepath, encoding="utf-8", errors="replace") as f:
                    for line_count, line in enumerate(f, 1):
                        if line_count <= old_lines:
                            continue
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        rtype = record.get("type")
                        if rtype not in ("assistant", "user", "custom-title", "ai-title"):
                            continue

                        session_id = record.get("sessionId")
                        if not session_id:
                            continue

                        # Extract session title from title records
                        title = _extract_title(record)
                        if title:
                            if session_id not in new_session_metas:
                                new_session_metas[session_id] = {
                                    "session_id": session_id,
                                    "project_name": "unknown",
                                    "first_timestamp": "",
                                    "last_timestamp": "",
                                    "git_branch": "",
                                    "model": None,
                                    "topic": None,
                                }
                            meta = new_session_metas[session_id]
                            if rtype == "custom-title":
                                meta["topic"] = title
                            elif rtype == "ai-title" and not meta.get("topic"):
                                meta["topic"] = title
                            continue

                        if rtype == "user":
                            dispatch = extract_agent_dispatch(record)
                            if dispatch is not None:
                                agents[dispatch["agent_id"]] = dispatch

                        timestamp = record.get("timestamp", "")
                        cwd = record.get("cwd", "")

                        # Track session metadata from new lines
                        if session_id not in new_session_metas:
                            new_session_metas[session_id] = {
                                "session_id": session_id,
                                "project_name": project_name_from_cwd(cwd),
                                "first_timestamp": timestamp,
                                "last_timestamp": timestamp,
                                "git_branch": record.get("gitBranch", ""),
                                "model": None,
                                "topic": None,
                            }
                        else:
                            meta = new_session_metas[session_id]
                            if timestamp and (not meta["last_timestamp"] or timestamp > meta["last_timestamp"]):
                                meta["last_timestamp"] = timestamp
                            if timestamp and (not meta["first_timestamp"] or timestamp < meta["first_timestamp"]):
                                meta["first_timestamp"] = timestamp

                        if rtype == "assistant":
                            msg = record.get("message", {})
                            usage = msg.get("usage", {})
                            model = msg.get("model", "")
                            message_id = msg.get("id", "")

                            input_tokens = usage.get("input_tokens", 0) or 0
                            output_tokens = usage.get("output_tokens", 0) or 0
                            cache_read = usage.get("cache_read_input_tokens", 0) or 0
                            cache_creation = usage.get("cache_creation_input_tokens", 0) or 0

                            if input_tokens + output_tokens + cache_read + cache_creation == 0:
                                continue

                            tool_name = None
                            for item in msg.get("content", []):
                                if isinstance(item, dict) and item.get("type") == "tool_use":
                                    tool_name = item.get("name")
                                    break

                            if model:
                                new_session_metas[session_id]["model"] = model

                            turn = {
                                "session_id": session_id,
                                "timestamp": timestamp,
                                "model": model,
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "cache_read_tokens": cache_read,
                                "cache_creation_tokens": cache_creation,
                                "tool_name": tool_name,
                                "cwd": cwd,
                                "message_id": message_id,
                                "is_subagent": 1 if is_subagent_record(record, filepath) else 0,
                                "agent_id": record_agent_id(record),
                            }

                            if message_id:
                                seen_messages[message_id] = turn
                            else:
                                turns_no_id.append(turn)
            except Exception as e:
                print(f"  Warning: {e}")

            if line_count <= old_lines:
                # File didn't grow (mtime changed but no new content)
                conn.execute("UPDATE processed_files SET mtime = ? WHERE path = ?",
                             (mtime, filepath))
                conn.commit()
                skipped_files += 1
                continue

            new_turns = turns_no_id + list(seen_messages.values())
            upsert_agents(conn, list(agents.values()))
            insert_skill_events(conn, parse_skill_events(filepath, skip_lines=old_lines))

            if new_turns or new_session_metas:
                sessions = aggregate_sessions(list(new_session_metas.values()), new_turns)
                upsert_sessions(conn, sessions)
                insert_turns(conn, new_turns)
                for s in sessions:
                    total_sessions.add(s["session_id"])
                total_turns += len(new_turns)
            updated_files += 1

        # Record file as processed (line_count already known from the single read)
        conn.execute("""
            INSERT OR REPLACE INTO processed_files (path, mtime, lines)
            VALUES (?, ?, ?)
        """, (filepath, mtime, line_count))
        conn.commit()

    # Recompute session totals from actual turns in DB.
    # This ensures correctness when INSERT OR IGNORE skips duplicate turns
    # but upsert_sessions had already added their tokens additively.
    if new_files or updated_files:
        conn.execute("""
            UPDATE sessions SET
                total_input_tokens = COALESCE((SELECT SUM(input_tokens) FROM turns WHERE turns.session_id = sessions.session_id), 0),
                total_output_tokens = COALESCE((SELECT SUM(output_tokens) FROM turns WHERE turns.session_id = sessions.session_id), 0),
                total_cache_read = COALESCE((SELECT SUM(cache_read_tokens) FROM turns WHERE turns.session_id = sessions.session_id), 0),
                total_cache_creation = COALESCE((SELECT SUM(cache_creation_tokens) FROM turns WHERE turns.session_id = sessions.session_id), 0),
                turn_count = COALESCE((SELECT COUNT(*) FROM turns WHERE turns.session_id = sessions.session_id), 0)
        """)
        conn.commit()

    upsert_agents(conn, _agents_from_meta(jsonl_files))
    _fill_unknown_projects(conn)
    conn.commit()

    # Copilot chats: on by default, off when the caller points the scan at a
    # custom transcripts dir (tests, --projects-dir) unless it passes dirs too.
    if copilot_dirs is None:
        copilot_dirs = COPILOT_DIRS if projects_dir is None and projects_dirs is None else []
    copilot_files = scan_copilot(conn, copilot_dirs, verbose)

    if verbose:
        print(f"\nScan complete:")
        print(f"  New files:     {new_files}")
        print(f"  Updated files: {updated_files}")
        print(f"  Skipped files: {skipped_files}")
        print(f"  Turns added:   {total_turns}")
        print(f"  Sessions seen: {len(total_sessions)}")
        print(f"  Copilot chats: {copilot_files} read")

    conn.close()
    return {"new": new_files, "updated": updated_files, "skipped": skipped_files,
            "turns": total_turns, "sessions": len(total_sessions), "copilot": copilot_files}


if __name__ == "__main__":
    import sys
    projects_dir = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--projects-dir" and i + 1 < len(sys.argv[1:]):
            projects_dir = Path(sys.argv[i + 2])
            break
    scan(projects_dir=projects_dir)
