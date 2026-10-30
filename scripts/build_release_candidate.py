"""Build an allowlisted release directory without modifying the workspace."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_CACHE_NAME = "models--sentence-transformers--all-MiniLM-L6-v2"
MODEL_FILES = (
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required release input missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def find_snapshot(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return hf_home / "hub" / MODEL_CACHE_NAME / "snapshots" / MODEL_REVISION


def verify_skill_master() -> tuple[str, str, str]:
    mapping_path = ROOT / "data" / "processed" / "job_skills_canonical.csv"
    frequency_path = ROOT / "data" / "processed" / "skill_frequency.csv"
    master_path = ROOT / "data" / "processed" / "skill_master.csv"
    mapping = pd.read_csv(mapping_path)[
        ["original_skill", "normalized_skill", "canonical_skill"]
    ].drop_duplicates()
    frequency = pd.read_csv(frequency_path)[
        ["canonical_skill", "skill_frequency", "jobs_with_skill", "job_percentage"]
    ]
    expected = mapping.merge(frequency, on="canonical_skill", how="left")

    def category(value):
        if pd.isna(value): return "unknown"
        if value >= 1000: return "very_common"
        if value >= 100: return "common"
        if value >= 10: return "moderate"
        if value >= 3: return "rare"
        return "very_rare"

    expected["frequency_category"] = expected["jobs_with_skill"].apply(category)
    expected = expected.sort_values(
        ["jobs_with_skill", "canonical_skill"], ascending=[False, True]
    )
    buffer = io.StringIO()
    expected.to_csv(buffer, index=False)
    if buffer.getvalue().encode("utf-8") != master_path.read_bytes():
        raise ValueError(
            "skill_master.csv does not match build_skill_master.py inputs; "
            "refusing to package an unverified runtime vocabulary."
        )
    return sha256(master_path), sha256(mapping_path), sha256(frequency_path)


def build(output: Path, frontend_dist: Path, embedding_snapshot: Path | None) -> Path:
    output = output.resolve()

    if output == ROOT or ROOT in output.parents:
        raise ValueError(
            "Release output must be outside the project repository."
        )

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing release directory: {output}")
    if not frontend_dist.is_dir():
        raise FileNotFoundError(f"Built frontend directory missing: {frontend_dist}")

    snapshot = find_snapshot(embedding_snapshot)

    if snapshot.name != MODEL_REVISION:
        raise ValueError(
            f"Embedding snapshot revision mismatch: expected {MODEL_REVISION}, "
            f"found {snapshot.name}"
        )

    missing = [name for name in MODEL_FILES if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Pinned local embedding snapshot is incomplete at {snapshot}: {missing}"
        )

    # Reject a build if the runtime vocabulary no longer matches its declared
    # canonical mapping/frequency inputs, without rebuilding project datasets.
    required = [
        ROOT / "backend" / "app",
        ROOT / "backend" / "migrations" / "versions",
        ROOT / "models" / "xgboost_ranker.json",
        ROOT / "models" / "ranking_features.txt",
        ROOT / "data" / "processed" / "skill_master.csv",
        ROOT / "requirements-runtime.lock",
    ]
    absent = [str(item.relative_to(ROOT)) for item in required if not item.exists()]
    if absent:
        raise FileNotFoundError(f"Required release inputs missing: {absent}")
    skill_hash, mapping_hash, frequency_hash = verify_skill_master()

    output.mkdir(parents=True)
    try:
        shutil.copytree(
            ROOT / "backend" / "app",
            output / "backend" / "app",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".env*"),
        )
        shutil.copytree(
            ROOT / "backend" / "migrations",
            output / "backend" / "migrations",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        copy_file(ROOT / "backend" / ".env.example", output / "backend" / ".env.example")
        copy_file(ROOT / "scripts" / "migrate_database.py", output / "scripts" / "migrate_database.py")
        copy_file(ROOT / "models" / "xgboost_ranker.json", output / "models" / "xgboost_ranker.json")
        copy_file(ROOT / "models" / "ranking_features.txt", output / "models" / "ranking_features.txt")
        copy_file(ROOT / "data" / "processed" / "skill_master.csv", output / "data" / "processed" / "skill_master.csv")
        copy_file(ROOT / "requirements-runtime.in", output / "requirements-runtime.in")
        copy_file(ROOT / "requirements-runtime.lock", output / "requirements-runtime.lock")
        copy_file(ROOT / "README.md", output / "README.md")
        copy_file(ROOT / "DEPLOYMENT.md", output / "DEPLOYMENT.md")
        shutil.copytree(frontend_dist.resolve(), output / "frontend" / "dist")
        for name in ("package.json", "package-lock.json"):
            copy_file(ROOT / "frontend" / name, output / "frontend" / name)

        cache_snapshot = (
            output / "hf-cache" / "hub" / MODEL_CACHE_NAME
            / "snapshots" / MODEL_REVISION
        )
        shutil.copytree(snapshot, cache_snapshot, symlinks=False)
        env_example = "\n".join((
            "APP_NAME=AI Resume Screening & Job Matching API",
            "APP_VERSION=1.0.0",
            "DEBUG=false",
            "DATABASE_URL=postgresql+psycopg://<user>:<password>@<host>:5432/<database>",
            "AUTH_SECRET=",
            "AUTH_TOKEN_TTL_MINUTES=720",
            "AUTH_COOKIE_SECURE=true",
            "UPLOAD_STORAGE_DIR=/var/lib/ai-resume-screening/uploads",
            f"EMBEDDING_MODEL_REVISION={MODEL_REVISION}",
            "EMBEDDING_LOCAL_FILES_ONLY=true",
            "HF_HOME=/app/hf-cache",
            "",
        ))
        (output / "deployment.env.example").write_text(env_example, encoding="utf-8")

        manifest = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "skill_master_sha256": skill_hash,
            "skill_master_sources": {
                "data/processed/job_skills_canonical.csv": mapping_hash,
                "data/processed/skill_frequency.csv": frequency_hash,
            },
            "files": {},
        }
        for path in sorted(p for p in output.rglob("*") if p.is_file()):
            manifest["files"][path.relative_to(output).as_posix()] = sha256(path)
        (output / "release-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        # Preserve partial output for inspection; never remove it automatically.
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frontend-dist", type=Path, default=ROOT / "frontend" / "dist")
    parser.add_argument("--embedding-snapshot", type=Path)
    args = parser.parse_args()
    result = build(args.output, args.frontend_dist, args.embedding_snapshot)
    print(f"Release candidate created: {result}")


if __name__ == "__main__":
    main()
