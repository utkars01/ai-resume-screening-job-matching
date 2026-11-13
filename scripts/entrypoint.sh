#!/bin/sh
set -e

echo "=== AI Resume Screening & Ranking Platform Starting ==="

if [ -n "$DATABASE_URL" ]; then
    echo "Running PostgreSQL schema migrations..."
    python scripts/migrate_database.py || echo "Warning: Database migration failed. Ensure PostgreSQL is running and accessible."
fi

PORT="${PORT:-8000}"
echo "Starting Uvicorn server on port ${PORT}..."
exec uvicorn backend.app.main:app --host 0.0.0.0 --port "$PORT" --workers 1 --proxy-headers --forwarded-allow-ips="*"
