from __future__ import annotations

import json
import os
import tempfile

import pytest
from optimum.intel import OVModelForFeatureExtraction
from optimum.onnxruntime import ORTModelForFeatureExtraction

from sentence_transformers import SentenceTransformer


## Testing exporting:
@pytest.mark.parametrize(
    ["backend", "expected_auto_model_class"],
    [
        ("onnx", ORTModelForFeatureExtraction),
        ("openvino", OVModelForFeatureExtraction),
    ],
)
@pytest.mark.parametrize(
    "model_kwargs", [{}, {"file_name": "wrong_file_name"}]
)  # <- Using a file_name is fine when exporting
def test_backend_export(backend, expected_auto_model_class, model_kwargs) -> None:
    model = SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors", backend=backend, model_kwargs=model_kwargs
    )
    assert model.get_backend() == backend
    assert isinstance(model[0].auto_model, expected_auto_model_class)
    embedding = model.encode("Hello, World!")
    assert embedding.shape == (model.get_sentence_embedding_dimension(),)


@pytest.mark.parametrize("backend", ["onnx", "openvino"])
def test_backend_no_export_crash(backend):
    with pytest.raises(OSError):
        SentenceTransformer(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors", backend=backend, model_kwargs={"export": False}
        )


## Testing loading exported models:
@pytest.mark.parametrize(
    ["backend", "model_id"],
    [
        ("onnx", "sentence-transformers-testing/stsb-bert-tiny-onnx"),
        ("openvino", "sentence-transformers-testing/stsb-bert-tiny-openvino"),
    ],
)
@pytest.mark.parametrize(
    ["model_kwargs", "exception"],
    [
        [{}, False],
        [{"file_name": "wrong_file_name", "export": True}, False],  # Using a file_name is fine when exporting
        [{"file_name": "wrong_file_name", "export": False}, True],  # ... but fails when not exporting
    ],
)
def test_backend_load(backend, model_id, model_kwargs, exception) -> None:
    if exception:
        with pytest.raises(OSError):
            SentenceTransformer(model_id, backend=backend, model_kwargs=model_kwargs)
    else:
        model = SentenceTransformer(model_id, backend=backend, model_kwargs=model_kwargs)
        assert model.get_backend() == backend
        embedding = model.encode("Hello, World!")
        assert embedding.shape == (model.get_sentence_embedding_dimension(),)


def test_onnx_provider_crash() -> None:
    with pytest.raises(ValueError):
        SentenceTransformer(
            "sentence-transformers-testing/stsb-bert-tiny-onnx",
            backend="onnx",
            model_kwargs={"provider": "incorrect_provider"},
        )


def test_openvino_provider() -> None:
    model = SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-openvino",
        backend="openvino",
        model_kwargs={"ov_config": {"INFERENCE_PRECISION_HINT": "precision_1"}},
    )
    assert model[0].auto_model.ov_config == {"INFERENCE_PRECISION_HINT": "precision_1", "PERFORMANCE_HINT": "LATENCY"}

    with tempfile.TemporaryDirectory() as temp_dir:
        ov_config_path = os.path.join(temp_dir, "ov_config.json")
        with open(ov_config_path, "w") as ov_config_file:
            json.dump({"INFERENCE_PRECISION_HINT": "precision_2"}, ov_config_file)

        model = SentenceTransformer(
            "sentence-transformers-testing/stsb-bert-tiny-openvino",
            backend="openvino",
            model_kwargs={"ov_config": ov_config_path},
        )
        assert model[0].auto_model.ov_config == {
            "INFERENCE_PRECISION_HINT": "precision_2",
            "PERFORMANCE_HINT": "LATENCY",
        }


def test_incorrect_backend() -> None:
    with pytest.raises(ValueError):
        SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors", backend="incorrect_backend")
