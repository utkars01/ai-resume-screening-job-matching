import logging
import sys
from types import ModuleType

import numpy as np
import pytest

from backend.app.services.embedding_service import EmbeddingService


def _install_fake_sentence_transformer(monkeypatch, model_type):
    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = model_type
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_embedding_service_logs_initialization_and_inference_once(
    monkeypatch, caplog
):
    class FakeModel:
        device = "cpu"
        init_calls = 0
        init_kwargs = None

        def __init__(self, name, **kwargs):
            self.name = name
            self.kwargs = kwargs
            type(self).init_calls += 1
            type(self).init_kwargs = kwargs

        def encode(self, text, normalize_embeddings):
            assert text == "synthetic resume"
            assert normalize_embeddings is True
            return np.asarray([0.6, 0.8], dtype=np.float32)

    _install_fake_sentence_transformer(monkeypatch, FakeModel)
    caplog.set_level(logging.DEBUG, logger="backend.app.services.embedding_service")
    service = EmbeddingService()

    assert service.generate_embedding("synthetic resume") == pytest.approx([0.6, 0.8])
    assert service.generate_embedding("synthetic resume") == pytest.approx([0.6, 0.8])

    initialized = [record for record in caplog.records if "model initialized" in record.message]
    inference = [record for record in caplog.records if "inference completed" in record.message]
    assert any("package import started" in record.message for record in caplog.records)
    assert any("model load started" in record.message for record in caplog.records)
    assert any("inference started" in record.message for record in caplog.records)
    assert len(initialized) == 1
    assert "import_seconds=" in initialized[0].message
    assert "load_seconds=" in initialized[0].message
    assert "device=cpu" in initialized[0].message
    assert len(inference) == 2
    assert "model_wait_seconds=" in inference[0].message
    assert "encode_seconds=" in inference[0].message
    assert "dimensions=(2,)" in inference[0].message
    assert FakeModel.init_calls == 1
    assert FakeModel.init_kwargs["revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert FakeModel.init_kwargs["local_files_only"] is True


def test_embedding_service_preload_uses_local_pinned_model(monkeypatch):
    class FakeModel:
        device = "cpu"

        def __init__(self, _name, **kwargs):
            assert kwargs["revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
            assert kwargs["local_files_only"] is True

    _install_fake_sentence_transformer(monkeypatch, FakeModel)
    service = EmbeddingService()
    assert not service.is_loaded
    service.preload()
    assert service.is_loaded


def test_embedding_service_logs_and_reraises_inference_failure(
    monkeypatch, caplog
):
    class FailingModel:
        device = "cpu"

        def __init__(self, _name, **_kwargs):
            pass

        def encode(self, *_args, **_kwargs):
            raise RuntimeError("synthetic inference failure")

    _install_fake_sentence_transformer(monkeypatch, FailingModel)
    caplog.set_level(logging.ERROR, logger="backend.app.services.embedding_service")

    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        EmbeddingService().generate_embedding("synthetic resume")

    failure = next(record for record in caplog.records if "inference failed" in record.message)
    assert "after " in failure.message
