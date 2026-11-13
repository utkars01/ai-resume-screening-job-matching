import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from backend.app.config import settings

from backend.app.api.auth import router as auth_router
from backend.app.api.jobs import router as jobs_router
from backend.app.api.applicants import router as applicants_router
from backend.app.database.session import get_db
from backend.app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app):
    # The image contains the pinned model in its Hugging Face cache. Load it
    # before accepting traffic so user requests never trigger downloads/init.
    await run_in_threadpool(embedding_service.preload)
    yield


app = FastAPI(
    title=settings.app_name,
    description="Backend API for the AI-powered recruitment platform.",
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
)

if settings.cors_origins:
    from fastapi.middleware.cors import CORSMiddleware

    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(auth_router)
app.include_router(jobs_router)
app.include_router(applicants_router)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/ready")
def readiness_check(db: Session = Depends(get_db)):
    if not embedding_service.is_loaded:
        raise HTTPException(status_code=503, detail="Embedding model is not ready.")
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        logger.warning("Readiness database check failed.")
        raise HTTPException(status_code=503, detail="Database is not ready.")
    return {"status": "ready"}


# Serve built frontend if available, otherwise fallback to API info on /
_frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if settings.serve_frontend and _frontend_dist.is_dir() and (_frontend_dist / "index.html").is_file():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {
            "message": settings.app_name,
            "version": settings.app_version,
            "status": "running",
        }

