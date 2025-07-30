"""
Test different backends (PyTorch, ONNX, OpenVINO) for the MLMTransformer class and SparseEncoder.

This module tests loading and using models with different inference backends.
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path

import pytest
from packaging.version import Version, parse

from tests.sparse_encoder.utils import sparse_allclose
from tests.utils import is_ci

try:
    from optimum.intel import OVModelForMaskedLM
    from optimum.intel.version import __version__ as optimum_intel_version
    from optimum.onnxruntime import ORTModelForMaskedLM
    from optimum.version import __version__ as optimum_version
except ImportError:
    pytest.skip("OpenVINO and ONNX backends are not available", allow_module_level=True)

from sentence_transformers.sparse_encoder import SparseEncoder

if is_ci():
    pytest.skip("Skip test in CI to try and avoid 429 Client Error", allow_module_level=True)


## Testing exporting:
@pytest.mark.parametrize(
    ["backend", "expected_auto_model_class"],
    [
        ("onnx", ORTModelForMaskedLM),
        ("openvino", OVModelForMaskedLM),
    ],
)
@pytest.mark.parametrize(
    "model_kwargs", [{}, {"file_name": "wrong_file_name"}]
)  # <- Using a file_name is fine when exporting
def test_backend_export(backend, expected_auto_model_class, model_kwargs) -> None:
    model = SparseEncoder("sparse-encoder-testing/splade-bert-tiny-nq", backend=backend, model_kwargs=model_kwargs)
    assert model.get_backend() == backend
    assert isinstance(model[0].auto_model, expected_auto_model_class)

    embedding = model.encode("Hello, World!")
    assert embedding.shape == (model.get_sentence_embedding_dimension(),)


def test_backend_no_export_crash():
    # Prior to optimum v1.25.0, ONNX Crashes when it can't export & the model repo/path doesn't contain an exported model
    # Since then, it auto-updates export to True
    with pytest.raises(OSError) if parse(optimum_version) < Version("1.25.0") else nullcontext():
        model = SparseEncoder(
            "sparse-encoder-testing/splade-bert-tiny-nq", backend="onnx", model_kwargs={"export": False}
        )
        assert isinstance(model[0].auto_model, ORTModelForMaskedLM)

    # OpenVINO will forcibly override the export=False if the model repo/path doesn't contain an exported model
    # But only starting from optimum-intel=v1.19.0
    with pytest.raises(OSError) if parse(optimum_intel_version) < Version("1.19.0") else nullcontext():
        model = SparseEncoder(
            "sparse-encoder-testing/splade-bert-tiny-nq",
            backend="openvino",
            model_kwargs={"export": False},
        )
        assert isinstance(model[0].auto_model, OVModelForMaskedLM)


## Testing loading exported models:
@pytest.mark.parametrize(
    ["backend", "model_id"],
    [
        ("onnx", "sparse-encoder-testing/splade-bert-tiny-nq-onnx"),
        ("openvino", "sparse-encoder-testing/splade-bert-tiny-nq-openvino"),
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
        with pytest.raises((OSError, RuntimeError)):
            SparseEncoder(model_id, backend=backend, model_kwargs=model_kwargs)
    else:
        model = SparseEncoder(model_id, backend=backend, model_kwargs=model_kwargs)
        assert model.get_backend() == backend
        tokens = model.encode("Hello, World!")
        assert len(tokens) > 0


def test_onnx_provider_crash() -> None:
    with pytest.raises(ValueError):
        SparseEncoder(
            "sparse-encoder-testing/splade-bert-tiny-nq-onnx",
            backend="onnx",
            model_kwargs={"provider": "incorrect_provider"},
        )


def test_openvino_provider() -> None:
    model = SparseEncoder(
        "sparse-encoder-testing/splade-bert-tiny-nq-openvino",
        backend="openvino",
        model_kwargs={"ov_config": {"INFERENCE_PRECISION_HINT": "precision_1"}},
    )
    assert model[0].auto_model.ov_config == {
        "INFERENCE_PRECISION_HINT": "precision_1",
        "PERFORMANCE_HINT": "LATENCY",
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        ov_config_path = os.path.join(temp_dir, "ov_config.json")
        with open(ov_config_path, "w") as ov_config_file:
            json.dump({"INFERENCE_PRECISION_HINT": "precision_2"}, ov_config_file)

        model = SparseEncoder(
            "sparse-encoder-testing/splade-bert-tiny-nq-openvino",
            backend="openvino",
            model_kwargs={"ov_config": ov_config_path},
        )
        assert model[0].auto_model.ov_config == {
            "INFERENCE_PRECISION_HINT": "precision_2",
            "PERFORMANCE_HINT": "LATENCY",
        }


def test_incorrect_backend() -> None:
    with pytest.raises(ValueError):
        SparseEncoder("sparse-encoder-testing/splade-bert-tiny-nq", backend="incorrect_backend")


def test_openvino_backend() -> None:
    model_id = "sparse-encoder-testing/splade-bert-tiny-nq"
    # Test that OpenVINO output is close to PyTorch output
    pytorch_model = SparseEncoder(model_id)
    openvino_model = SparseEncoder(
        model_id,
        backend="openvino",
        model_kwargs={"ov_config": {"INFERENCE_PRECISION_HINT": "f32"}},
    )
    pytorch_result = pytorch_model.encode(["Hello there!"], save_to_cpu=True)
    openvino_result = openvino_model.encode(["Hello there!"])
    assert sparse_allclose(openvino_result, pytorch_result, atol=0.00001), "OpenVINO and Pytorch outputs are not close"

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Test that loading with ov_config file works as expected
        config_file = str(Path(tmpdirname) / "ov_config.json")
        with open(Path(config_file), "w") as f:
            f.write('{"NUM_STREAMS" : "2"}')
        openvino_model_with_config = SparseEncoder(
            model_id,
            backend="openvino",
            model_kwargs={"ov_config": config_file},
        )
        # The transformers model is an Optimum model with an OpenVINO inference request property
        assert openvino_model_with_config[0].auto_model.request.get_property("NUM_STREAMS") == 2

        # Test that saving and loading local OpenVINO models works as expected
        openvino_model_with_config.save_pretrained(tmpdirname)
        local_openvino_model = SparseEncoder(
            tmpdirname, backend="openvino", model_kwargs={"ov_config": {"INFERENCE_PRECISION_HINT": "f32"}}
        )
        local_openvino_result = local_openvino_model.encode(["Hello there!"])
        assert sparse_allclose(
            local_openvino_result, openvino_result
        ), "OpenVINO saved model output differs from in-memory converted model"
        del local_openvino_model
        gc.collect()


def test_export_false_subfolder() -> None:
    model_id = "sparse-encoder-testing/splade-bert-tiny-nq-openvino"

    def from_pretrained_decorator(method):
        def decorator(*args, **kwargs):
            assert not kwargs["export"]
            assert kwargs["subfolder"] == "openvino"
            assert kwargs["file_name"] == "openvino_model.xml"
            return method(*args, **kwargs)

        return decorator

    OVModelForMaskedLM.from_pretrained = from_pretrained_decorator(OVModelForMaskedLM.from_pretrained)
    SparseEncoder(model_id, backend="openvino", model_kwargs={"export": False})


def test_export_set_nested_filename() -> None:
    model_id = "sparse-encoder-testing/splade-bert-tiny-nq-openvino"

    def from_pretrained_decorator(method):
        def decorator(*args, **kwargs):
            assert kwargs["subfolder"] == "openvino"
            assert kwargs["file_name"] == "openvino_model.xml"
            return method(*args, **kwargs)

        return decorator

    OVModelForMaskedLM.from_pretrained = from_pretrained_decorator(OVModelForMaskedLM.from_pretrained)
    SparseEncoder(model_id, backend="openvino", model_kwargs={"file_name": "openvino/openvino_model.xml"})
