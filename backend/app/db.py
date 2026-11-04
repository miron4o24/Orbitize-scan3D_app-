"""Таблици за профилите (app/accounts.py): потребители, техните сесии, кои сканирания са техни, харесвания и сигнали в
общността (app/community.py), спечелените точки (app/progress.py) и настройки на сървъра, които могат да се променят
без достъп до него (идентификаторът на клиента за Google). MySQL, когато в .env е зададено MYSQL_HOST (или
DATABASE_URL); дотогава файл на SQLite до задачите (accounts.db), така че входът работи и без сървър за бази данни.
Първото стартиране с MySQL прехвърля профилите от този файл. DATABASE_URL=off изключва профилите: остава само
споделеният API ключ. База данни, която не работи при стартиране на сървъра, се опитва отново при следващата заявка."""

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
    sa.Column("google_sub", sa.String(64), nullable=False, unique=True),  # идентификаторът на профила в Google
    sa.Column("email", sa.String(320), nullable=False),
    sa.Column("name", sa.String(100), nullable=False),
    sa.Column("google_picture", sa.String(1024)),
    sa.Column("avatar", sa.Boolean, nullable=False, default=False),  # качена е собствена снимка
    sa.Column("created", sa.BigInteger, nullable=False),  # секунди по Unix, като "created" при задачите
    sa.Column("password_hash", sa.String(200)),  # профили с имейл (accounts.hash_password); google_sub "email:<имейл>"
    # Pro: AI моделите (app/ai.py). Включва се от собственика в приложението (accounts.OWNER_EMAILS); по-късно чрез
    # плащане.
    sa.Column("pro", sa.Boolean, nullable=False, default=False, server_default=sa.false()),
    sa.Column("pro_until", sa.BigInteger),  # Pro чрез плащане: докога важи според Google (app/billing.py)
    sa.Column("pro_token", sa.String(512)),  # ключът за покупка на абонамента, за да се пита Google отново в края му
)

# Колони, добавени след първото пускане на таблиците: create_all създава само липсващи таблици, а не липсващи колони.
LATER_COLUMNS = {
    "users": {"password_hash": "VARCHAR(200)", "pro": "BOOLEAN NOT NULL DEFAULT 0", "pro_until": "BIGINT",
              "pro_token": "VARCHAR(512)"},
    "scans": {"published": "BIGINT", "longest_mm": "FLOAT"},
}

sessions = sa.Table(
    "sessions",
    meta,
    sa.Column("token_hash", sa.String(64), primary_key=True),  # SHA-256 на ключа: изтекла таблица не дава достъп на никого
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
    sa.Column("share", sa.String(32), unique=True),  # номерът на публичния линк, докато е споделено
    sa.Column("published", sa.BigInteger),  # секунди по Unix, откакто е в общността (app/community.py); None: не е в нея
    sa.Column("longest_mm", sa.Float),  # реалният размер, зададен от автора при публикуване
)

# Общността (app/community.py): по един ред за всеки човек, харесал или сигнализирал за публикация.
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

# Точки (app/progress.py): по един ред за всяко спечелило нещо, веднъж за потребител, вид и ref.
xp_events = sa.Table(
    "xp_events",
    meta,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
    sa.Column("kind", sa.String(20), nullable=False),  # scan, publish, liked, stress, challenge
    sa.Column("ref", sa.String(80), nullable=False),  # какво ги е спечелило: номер на задача, job:liker, job:case, week:challenge
    sa.Column("points", sa.Integer, nullable=False),
    sa.Column("created", sa.BigInteger, nullable=False),  # секунди по Unix, когато се е случило
    sa.UniqueConstraint("user_id", "kind", "ref"),
)

# Броените използвания от безплатния план (app/plans.py): по един ред за профил, вид ("stress", "ar") и обект (номер
# на задачата на модела), така че повторна употреба на същия модел се брои веднъж.
plan_uses = sa.Table(
    "plan_uses",
    meta,
    sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    sa.Column("kind", sa.String(20), primary_key=True),
    sa.Column("ref", sa.String(80), primary_key=True),
    sa.Column("created", sa.BigInteger, nullable=False),
)

push_tokens = sa.Table(  # телефони, които да се известяват, когато модел е готов (app/notify.py)
    "push_tokens",
    meta,
    sa.Column("token", sa.String(200), primary_key=True),
    sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
    sa.Column("created", sa.BigInteger, nullable=False),
)

purchases = sa.Table(  # на кой профил служи всяка покупка в Google Play (app/billing.py)
    "purchases",
    meta,
    sa.Column("token_hash", sa.String(64), primary_key=True),
    sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
    sa.Column("created", sa.BigInteger, nullable=False),
)

prints = sa.Table(  # снимки на модел от общността, отпечатан от някого (app/community.py); по една на човек и модел
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
    """Свързаният механизъм със създадени таблици или None (изключен или не работи: вижте status())."""
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
            except Exception as ex:  # грешна парола, сървърът не работи, няма такава база данни...
                _error = f"{type(ex).__name__}: {ex}".splitlines()[0][:300]
                log.error("database: %s", _error)
    return _engine


def connect(u: str | sa.URL) -> sa.Engine:
    if sa.make_url(u).get_backend_name() != "sqlite":
        # pre_ping: MySQL затваря връзки, неизползвани 8 часа; recycle ги поддържа по-млади от това.
        return sa.create_engine(u, pool_pre_ping=True, pool_recycle=3600)
    # Заявките вървят в няколко нишки; timeout: изчаква записа на друга нишка, вместо да се провали веднага.
    e = sa.create_engine(u, connect_args={"check_same_thread": False, "timeout": 15})

    @sa.event.listens_for(e, "connect")
    def _foreign_keys(conn, _record):  # SQLite пренебрегва ON DELETE CASCADE без това
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
    """Копира веднъж профилите от файла на SQLite в празна база данни на MySQL (след това файлът се преименува)."""
    path = sqlite_file()
    if not path.exists():
        return
    with e.connect() as c:
        if c.execute(sa.select(sa.func.count()).select_from(users)).scalar():
            return
    old = connect(sa.URL.create("sqlite", database=str(path)))
    try:
        meta.create_all(old)  # файл отпреди добавянето на таблица: празна, така че няма какво да се копира, но може да се чете
        add_later_columns(old)
        with old.connect() as src, e.begin() as dst:
            for table in meta.sorted_tables:  # потребителите преди таблиците, които сочат към тях
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
    """Тестове: забравя механизма, за да се свърже следващото извикване с текущия DATABASE_URL."""
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
    # Не "update, insert при 0 реда": MySQL брои обновяване със същата стойност като 0 променени реда.
    with engine().begin() as c:
        if c.execute(sa.select(settings.c.name).where(settings.c.name == name)).first():
            c.execute(sa.update(settings).where(settings.c.name == name).values(value=value))
        else:
            c.execute(sa.insert(settings).values(name=name, value=value))
