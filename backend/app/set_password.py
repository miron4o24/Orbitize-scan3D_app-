"""Нова парола на профил с имейл, от командния ред на сървъра (няма го в програмния интерфейс): python -m
app.set_password <имейл> <парола>. Паролата минава същото правило като при регистрация (accounts.password_ok), а
профилът излиза от всички телефони."""

import sys

import sqlalchemy as sa

from . import accounts as A
from . import db


def set_password(email: str, password: str) -> str:
    if not A.password_ok(password):
        raise SystemExit(A.PASSWORD_RULE)
    engine = db.engine()
    if engine is None:
        raise SystemExit("Профилите са изключени (DATABASE_URL=off).")
    email = email.strip().lower()
    with engine.begin() as c:
        row = c.execute(sa.select(db.users.c.id).where(sa.func.lower(db.users.c.email) == email)).first()
        if row is None:
            raise SystemExit(f"Няма профил с имейл {email}.")
        c.execute(sa.update(db.users).where(db.users.c.id == row.id).values(password_hash=A.hash_password(password)))
        c.execute(sa.delete(db.sessions).where(db.sessions.c.user_id == row.id))
    return row.id


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Употреба: python -m app.set_password <имейл> <парола>")
    set_password(sys.argv[1], sys.argv[2])
    print(f"Паролата на {sys.argv[1]} е сменена; профилът излезе от всички телефони.")
