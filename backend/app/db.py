"""Tables for accounts (app/accounts.py): users, their sign-in sessions, which scans are theirs, the community's likes
and reports (app/community.py), the points they earned (app/progress.py), and server settings that can be changed without touching the server (the Google
client ID).

MySQL when MYSQL_HOST (or DATABASE_URL) is set in .env; until then a SQLite file next to the jobs (accounts.db), so
sign-in works without setting up a database server. The first start on MySQL moves that file's accounts over.
DATABASE_URL=off turns accounts off: the shared API token only. A database that is down when the server starts is
tried again on the next request."""

import logging
import os
import threading
import time
from pathlib import Path

import sqlalchemy as sa

log = logging.getLogger("uvicorn.error")

meta = sa.MetaData()

users = sa.Table(
    "users",
    meta,
    sa.Column("id", sa.String(32), primary_key=True),
    sa.Column("google_sub", sa.String(64), nullable=False, unique=True),  # Google's id for the account
    sa.Column("email", sa.String(320), nullable=False),
    sa.Column("name", sa.String(100), nullable=False),
    sa.Column("google_picture", sa.String(1024)),
    sa.Column("avatar", sa.Boolean, nullable=False, default=False),  # an own photo was uploaded
    sa.Column("created", sa.BigInteger, nullable=False),  # Unix seconds, like the jobs' "created"
    sa.Column("password_hash", sa.String(200)),  # email accounts (accounts.hash_password); google_sub "email:<email>"
    # Pro: the AI models (app/ai.py). Switched on by the owner in the app (accounts.OWNER_EMAILS); later by paying.
    sa.Column("pro", sa.Boolean, nullable=False, default=False, server_default=sa.false()),
    sa.Column("pro_until", sa.BigInteger),  # Pro by payment: until when Google says it runs (app/billing.py)
    sa.Column("pro_token", sa.String(512)),  # that subscription's purchase token, to ask Google again at its end
)

# Columns added after the tables first went out: create_all makes missing tables only, not missing columns.
LATER_COLUMNS = {
    "users": {"password_hash": "VARCHAR(200)", "pro": "BOOLEAN NOT NULL DEFAULT 0", "pro_until": "BIGINT",
              "pro_token": "VARCHAR(512)"},
    "scans": {"published": "BIGINT", "longest_mm": "FLOAT"},
}

sessions = sa.Table(
    "sessions",
    meta,
    sa.Column("token_hash", sa.String(64), primary_key=True),  # SHA-256 of the token: a leaked table signs no one in
    sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
    sa.Column("created", sa.BigInteger, nullable=False),
    sa.Column("last_used", sa.BigInteger, nullable=False),
)

scans = sa.Table(
    "scans",
    meta,
    sa.Column("job_id", sa.String(32), primary_key=True),
    sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
    sa.Column("title", sa.String(100), nullable=False, default=""),
    sa.Column("created", sa.BigInteger, nullable=False),
    sa.Column("share", sa.String(32), unique=True),  # the public link's id while shared
    sa.Column("published", sa.BigInteger),  # Unix seconds it went into the community (app/community.py); None: not in it
    sa.Column("longest_mm", sa.Float),  # the real size its author gave when posting it
)

# The community (app/community.py): one row per person who liked a post, and per person who reported one.
likes = sa.Table(
    "likes",
    meta,
    sa.Column("job_id", sa.String(32), sa.ForeignKey("scans.job_id", ondelete="CASCADE"), primary_key=True),
    sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, index=True),
    sa.Column("created", sa.BigInteger, nullable=False),
)

reports = sa.Table(
    "reports",
    meta,
    sa.Column("job_id", sa.String(32), sa.ForeignKey("scans.job_id", ondelete="CASCADE"), primary_key=True),
    sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, index=True),
    sa.Column("reason", sa.String(200), nullable=False, default=""),
    sa.Column("created", sa.BigInteger, nullable=False),
)

# Points (app/progress.py): one row per thing that earned some, once per user, kind and ref.
xp_events = sa.Table(
    "xp_events",
    meta,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
    sa.Column("kind", sa.String(20), nullable=False),  # scan, publish, liked, stress, challenge
    sa.Column("ref", sa.String(80), nullable=False),  # what earned it: a job id, job:liker, job:case, week:challenge
    sa.Column("points", sa.Integer, nullable=False),
    sa.Column("created", sa.BigInteger, nullable=False),  # Unix seconds it happened
    sa.UniqueConstraint("user_id", "kind", "ref"),
)

# The free plan's counted uses (app/plans.py): one row per account, kind ("stress", "ar") and thing (the model's job
# id), so using the same model again counts once.
plan_uses = sa.Table(
    "plan_uses",
    meta,
    sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    sa.Column("kind", sa.String(20), primary_key=True),
    sa.Column("ref", sa.String(80), primary_key=True),
    sa.Column("created", sa.BigInteger, nullable=False),
)

push_tokens = sa.Table(  # phones to tell when a model is ready (app/notify.py)
    "push_tokens",
    meta,
    sa.Column("token", sa.String(200), primary_key=True),
    sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
    sa.Column("created", sa.BigInteger, nullable=False),
)

purchases = sa.Table(  # which account each Google Play purchase serves (app/billing.py)
    "purchases",
    meta,
    sa.Column("token_hash", sa.String(64), primary_key=True),
    sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
    sa.Column("created", sa.BigInteger, nullable=False),
)

prints = sa.Table(  # photos of a community model printed by someone (app/community.py); one per person and model
    "prints",
    meta,
    sa.Column("id", sa.String(32), primary_key=True),
    sa.Column("job_id", sa.String(32), sa.ForeignKey("scans.job_id", ondelete="CASCADE"), nullable=False, index=True),
    sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
    sa.Column("created", sa.BigInteger, nullable=False),
)

print_reports = sa.Table(
    "print_reports",
    meta,
    sa.Column("print_id", sa.String(32), sa.ForeignKey("prints.id", ondelete="CASCADE"), primary_key=True),
    sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    sa.Column("created", sa.BigInteger, nullable=False),
)

settings = sa.Table(
    "settings",
    meta,
    sa.Column("name", sa.String(64), primary_key=True),
    sa.Column("value", sa.Text, nullable=False),
)


def sqlite_file() -> Path:
    from . import jobs

    return Path(os.environ.get("ACCOUNTS_DB", jobs.DATA.parent / "accounts.db")).resolve()


def url() -> str | sa.URL | None:
    if u := os.environ.get("DATABASE_URL"):
        return None if u.lower() == "off" else u
    host = os.environ.get("MYSQL_HOST")
    if not host:
        return sa.URL.create("sqlite", database=str(sqlite_file()))
    return sa.URL.create(
        "mysql+pymysql",
        username=os.environ.get("MYSQL_USER", "scan3d"),
        password=os.environ.get("MYSQL_PASSWORD", ""),
        host=host,
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        database=os.environ.get("MYSQL_DATABASE", "scan3d"),
        query={"charset": "utf8mb4"},
    )


_lock = threading.Lock()
_engine: sa.Engine | None = None
_error: str | None = None
_tried = 0.0
RETRY_S = 30


def configured() -> bool:
    return url() is not None


def engine() -> sa.Engine | None:
    """The connected engine with the tables made, or None (off, or down: see status())."""
    global _engine, _error, _tried
    if _engine is not None or not configured():
        return _engine
    with _lock:
        if _engine is None and time.time() - _tried > RETRY_S:
            _tried = time.time()
            try:
                e = connect(url())
                meta.create_all(e)
                add_later_columns(e)
                if e.dialect.name != "sqlite":
                    move_from_sqlite(e)
                _engine, _error = e, None
                log.info("database: %s connected, tables ready", e.dialect.name)
            except Exception as ex:  # wrong password, server down, no such database...
                _error = f"{type(ex).__name__}: {ex}".splitlines()[0][:300]
                log.error("database: %s", _error)
    return _engine


def connect(u: str | sa.URL) -> sa.Engine:
    if sa.make_url(u).get_backend_name() != "sqlite":
        # pre_ping: MySQL closes connections idle for 8 hours; recycle keeps them younger than that.
        return sa.create_engine(u, pool_pre_ping=True, pool_recycle=3600)
    # Requests run on several threads; timeout: wait for another thread's write instead of failing at once.
    e = sa.create_engine(u, connect_args={"check_same_thread": False, "timeout": 15})

    @sa.event.listens_for(e, "connect")
    def _foreign_keys(conn, _record):  # SQLite ignores ON DELETE CASCADE without this
        conn.execute("PRAGMA foreign_keys = ON")

    return e


def add_later_columns(e: sa.Engine):
    for table, columns in LATER_COLUMNS.items():
        have = {c["name"] for c in sa.inspect(e).get_columns(table)}
        with e.begin() as c:
            for name, kind in columns.items():
                if name not in have:
                    c.execute(sa.text(f"ALTER TABLE {table} ADD COLUMN {name} {kind}"))
                    log.info("database: added %s.%s", table, name)


def move_from_sqlite(e: sa.Engine):
    """Copies the SQLite file's accounts into an empty MySQL database, once (the file is renamed after)."""
    path = sqlite_file()
    if not path.exists():
        return
    with e.connect() as c:
        if c.execute(sa.select(sa.func.count()).select_from(users)).scalar():
            return
    old = connect(sa.URL.create("sqlite", database=str(path)))
    try:
        meta.create_all(old)  # a file from before a table was added: empty, so nothing to copy, but selectable
        add_later_columns(old)
        with old.connect() as src, e.begin() as dst:
            for table in meta.sorted_tables:  # users before the tables that point at them
                rows = [dict(r._mapping) for r in src.execute(sa.select(table))]
                if rows:
                    dst.execute(sa.insert(table), rows)
                log.info("database: moved %d rows of %s from %s", len(rows), table.name, path.name)
    finally:
        old.dispose()
    path.rename(path.with_name(path.name + ".moved"))


def status() -> str:
    if not configured():
        return "off"
    return "ok" if engine() is not None else f"error: {_error}"


def reset():
    """Tests: forget the engine so the next call connects to the current DATABASE_URL."""
    global _engine, _error, _tried
    if _engine is not None:
        _engine.dispose()
    _engine, _error, _tried = None, None, 0.0


def get_setting(name: str) -> str | None:
    e = engine()
    if e is None:
        return None
    with e.connect() as c:
        return c.execute(sa.select(settings.c.value).where(settings.c.name == name)).scalar()


def set_setting(name: str, value: str):
    # Not "update, insert if 0 rows": MySQL counts an update to the same value as 0 rows changed.
    with engine().begin() as c:
        if c.execute(sa.select(settings.c.name).where(settings.c.name == name)).first():
            c.execute(sa.update(settings).where(settings.c.name == name).values(value=value))
        else:
            c.execute(sa.insert(settings).values(name=name, value=value))
