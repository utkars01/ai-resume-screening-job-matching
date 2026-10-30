"""Apply ordered, checksummed PostgreSQL migrations to a fresh database.

Existing unversioned databases are deliberately rejected. They need a reviewed
baseline adoption process before this runner can manage them.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from sqlalchemy import create_engine, text


ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "backend" / "migrations" / "versions"
VERSION_FILE = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


def apply_migrations(database_url: str) -> list[str]:
    engine = create_engine(database_url, pool_pre_ping=True)
    applied_now: list[str] = []
    try:
        with engine.begin() as connection:
            if connection.dialect.name != "postgresql":
                raise RuntimeError("Migrations require PostgreSQL.")

            connection.execute(text("SELECT pg_advisory_xact_lock(4815162342)"))
            tables = set(connection.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = current_schema()"
            )).scalars())

            if "schema_migrations" not in tables and tables:
                raise RuntimeError(
                    "Refusing to migrate an existing unversioned schema. "
                    "Review and adopt its baseline explicitly first."
                )

            connection.execute(text(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version VARCHAR(255) PRIMARY KEY, "
                "checksum CHAR(64) NOT NULL, "
                "applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            ))
            applied = dict(connection.execute(text(
                "SELECT version, checksum FROM schema_migrations ORDER BY version"
            )).all())
            if tables - {"schema_migrations"} and not applied:
                raise RuntimeError(
                    "Refusing to migrate a populated schema with empty migration history. "
                    "Review and adopt its baseline explicitly first."
                )

            migration_files = sorted(
                path for path in MIGRATIONS.iterdir()
                if VERSION_FILE.fullmatch(path.name)
            )
            versions = [VERSION_FILE.fullmatch(path.name).group(1) for path in migration_files]
            if len(versions) != len(set(versions)):
                raise RuntimeError("Duplicate migration version numbers.")

            expected_prefix: list[str] = []
            for path, version in zip(migration_files, versions):
                sql = path.read_text(encoding="utf-8")
                checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
                expected_prefix.append(version)
                if version in applied:
                    if applied[version].strip() != checksum:
                        raise RuntimeError(f"Applied migration checksum changed: {path.name}")
                    continue
                if any(applied_version not in expected_prefix for applied_version in applied):
                    raise RuntimeError("Migration history is out of order or unknown.")
                connection.exec_driver_sql(sql)
                connection.execute(text(
                    "INSERT INTO schema_migrations (version, checksum) "
                    "VALUES (:version, :checksum)"
                ), {"version": version, "checksum": checksum})
                applied_now.append(version)

            unknown = set(applied) - set(versions)
            if unknown:
                raise RuntimeError(f"Unknown applied migration versions: {sorted(unknown)}")
    finally:
        engine.dispose()
    return applied_now


def main() -> None:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("Set DATABASE_URL for the intended PostgreSQL database.")
    applied = apply_migrations(database_url)
    if applied:
        print("Applied migrations: " + ", ".join(applied))
    else:
        print("Database schema is already up to date.")


if __name__ == "__main__":
    main()
