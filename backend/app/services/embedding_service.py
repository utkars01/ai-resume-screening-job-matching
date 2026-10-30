import logging
from time import perf_counter
from threading import Lock

from backend.app.config import settings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self._model = None
        self._model_lock = Lock()

    def _get_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    # Import and load the large model only when embedding is
                    # actually requested. API startup and unit tests should
                    # not trigger model downloads or initialization.
                    started = perf_counter()
                    try:
                        import_started = perf_counter()
                        logger.info(
                            "Sentence embedding package import started "
                            "(model=%s)",
                            MODEL_NAME,
                        )
                        from sentence_transformers import SentenceTransformer
                        import_seconds = perf_counter() - import_started

                        load_started = perf_counter()
                        logger.info(
                            "Sentence embedding model load started "
                            "(model=%s)",
                            MODEL_NAME,
                        )
                        model = SentenceTransformer(
                            MODEL_NAME,
                            revision=settings.embedding_model_revision,
                            local_files_only=settings.embedding_local_files_only,
                        )
                        load_seconds = perf_counter() - load_started
                    except Exception:
                        logger.exception(
                            "Sentence embedding model initialization failed "
                            "after %.3f seconds (model=%s)",
                            perf_counter() - started,
                            MODEL_NAME,
                        )
                        raise

                    self._model = model
                    logger.info(
                        "Sentence embedding model initialized "
                        "(model=%s, import_seconds=%.3f, "
                        "load_seconds=%.3f, total_seconds=%.3f, device=%s)",
                        MODEL_NAME,
                        import_seconds,
                        load_seconds,
                        perf_counter() - started,
                        getattr(model, "device", "unknown"),
                    )
        return self._model

    def preload(self):
        """Load the pinned model from the local cache before serving requests."""
        self._get_model()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def generate_embedding(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        model_started = perf_counter()
        model = self._get_model()
        model_wait_seconds = perf_counter() - model_started

        inference_started = perf_counter()
        logger.debug(
            "Sentence embedding inference started "
            "(model=%s, text_characters=%d)",
            MODEL_NAME,
            len(text),
        )
        try:
            embedding = model.encode(
                text,
                normalize_embeddings=True,
            )
        except Exception:
            logger.exception(
                "Sentence embedding inference failed after %.3f seconds "
                "(model=%s)",
                perf_counter() - inference_started,
                MODEL_NAME,
            )
            raise

        logger.debug(
            "Sentence embedding inference completed "
            "(model=%s, model_wait_seconds=%.3f, encode_seconds=%.3f, "
            "dimensions=%s)",
            MODEL_NAME,
            model_wait_seconds,
            perf_counter() - inference_started,
            getattr(embedding, "shape", (len(embedding),)),
        )

        return embedding.tolist()


embedding_service = EmbeddingService()
