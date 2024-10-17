"""
Tests general behaviour of the SentenceTransformer class
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import partial
from pathlib import Path
from typing import Dict, List, Literal, cast

import numpy as np
import pytest
import torch
from huggingface_hub import CommitInfo, HfApi, RepoUrl
from torch import nn

from sentence_transformers import SentenceTransformer, util
from sentence_transformers.models import (
    CNN,
    LSTM,
    Dense,
    LayerNorm,
    Normalize,
    Pooling,
    Transformer,
    WeightedLayerPooling,
)
from sentence_transformers.similarity_functions import SimilarityFunction
from tests.utils import SafeTemporaryDirectory


def test_load_with_safetensors() -> None:
    with SafeTemporaryDirectory() as cache_folder:
        safetensors_model = SentenceTransformer(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors",
            cache_folder=cache_folder,
        )

        # Only the safetensors file must be loaded
        pytorch_files = list(Path(cache_folder).glob("**/pytorch_model.bin"))
        assert 0 == len(pytorch_files), "PyTorch model file must not be downloaded."
        safetensors_files = list(Path(cache_folder).glob("**/model.safetensors"))
        assert 1 == len(safetensors_files), "Safetensors model file must be downloaded."

    with SafeTemporaryDirectory() as cache_folder:
        transformer = Transformer(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors",
            cache_dir=cache_folder,
            model_args={"use_safetensors": False},
        )
        pooling = Pooling(transformer.get_word_embedding_dimension())
        pytorch_model = SentenceTransformer(modules=[transformer, pooling])

        # Only the pytorch file must be loaded
        pytorch_files = list(Path(cache_folder).glob("**/pytorch_model.bin"))
        assert 1 == len(pytorch_files), "PyTorch model file must be downloaded."
        safetensors_files = list(Path(cache_folder).glob("**/model.safetensors"))
        assert 0 == len(safetensors_files), "Safetensors model file must not be downloaded."

    sentences = ["This is a test sentence", "This is another test sentence"]
    assert torch.equal(
        safetensors_model.encode(sentences, convert_to_tensor=True),
        pytorch_model.encode(sentences, convert_to_tensor=True),
    ), "Ensure that Safetensors and PyTorch loaded models result in identical embeddings"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA must be available to test moving devices effectively.")
def test_to() -> None:
    model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors", device="cpu")

    test_device = torch.device("cuda")
    assert model.device.type == "cpu"
    assert test_device.type == "cuda"

    model.to(test_device)
    assert model.device.type == "cuda", "The model device should have updated"

    model.encode("Test sentence")
    assert model.device.type == "cuda", "Encoding shouldn't change the device"

    assert model._target_device == model.device, "Prevent backwards compatibility failure for _target_device"
    model._target_device = "cpu"
    assert model.device.type == "cpu", "Ensure that setting `_target_device` doesn't crash."


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA must be available to test fp16 and bf16 inference.")
@pytest.mark.parametrize("torch_dtype", ["auto", "float16", "bfloat16", torch.float16, torch.bfloat16])
def test_torch_dtype(torch_dtype) -> None:
    model = SentenceTransformer(
        "sentence-transformers-testing/all-nli-bert-tiny-dense",
        device="cuda",
        model_kwargs={"torch_dtype": torch_dtype},
    )
    embedding = model.encode("Test sentence")
    assert embedding.shape[-1] == model.get_sentence_embedding_dimension()


def test_push_to_hub(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def mock_create_repo(self, repo_id, **kwargs):
        return RepoUrl(f"https://huggingface.co/{repo_id}")

    mock_upload_folder_kwargs = {}

    def mock_upload_folder(self, **kwargs):
        nonlocal mock_upload_folder_kwargs
        mock_upload_folder_kwargs = kwargs
        if kwargs.get("revision") is None:
            return CommitInfo(
                commit_url=f"https://huggingface.co/{kwargs.get('repo_id')}/commit/123456",
                commit_message="commit_message",
                commit_description="commit_description",
                oid="oid",
            )
        else:
            return CommitInfo(
                commit_url=f"https://huggingface.co/{kwargs.get('repo_id')}/commit/678901",
                commit_message="commit_message",
                commit_description="commit_description",
                oid="oid",
            )

    def mock_create_branch(self, repo_id, branch, revision=None, **kwargs):
        return None

    monkeypatch.setattr(HfApi, "create_repo", mock_create_repo)
    monkeypatch.setattr(HfApi, "upload_folder", mock_upload_folder)
    monkeypatch.setattr(HfApi, "create_branch", mock_create_branch)

    model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")

    url = model.push_to_hub("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    assert mock_upload_folder_kwargs["repo_id"] == "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    assert url == "https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors/commit/123456"
    mock_upload_folder_kwargs.clear()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        url = model.save_to_hub("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        assert mock_upload_folder_kwargs["repo_id"] == "sentence-transformers-testing/stsb-bert-tiny-safetensors"
        assert url == "https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors/commit/123456"
        mock_upload_folder_kwargs.clear()
        assert len(caplog.record_tuples) == 1
        assert (
            caplog.record_tuples[0][2]
            == "The `save_to_hub` method is deprecated and will be removed in a future version of SentenceTransformers. Please use `push_to_hub` instead for future model uploads."
        )

    with pytest.raises(
        ValueError, match="Providing an `organization` to `save_to_hub` is deprecated, please only use `repo_id`."
    ):
        model.save_to_hub("sentence-transformers-testing/stsb-bert-tiny-safetensors", organization="unrelated")

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        url = model.save_to_hub(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors", organization="sentence-transformers-testing"
        )
        assert mock_upload_folder_kwargs["repo_id"] == "sentence-transformers-testing/stsb-bert-tiny-safetensors"
        assert url == "https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors/commit/123456"
        assert len(caplog.record_tuples) == 2
        assert (
            caplog.record_tuples[0][2]
            == "The `save_to_hub` method is deprecated and will be removed in a future version of SentenceTransformers. Please use `push_to_hub` instead for future model uploads."
        )
        assert (
            caplog.record_tuples[1][2]
            == 'Providing an `organization` to `save_to_hub` is deprecated, please only use `repo_id="sentence-transformers-testing/stsb-bert-tiny-safetensors"` instead.'
        )
    mock_upload_folder_kwargs.clear()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        url = model.save_to_hub("stsb-bert-tiny-safetensors", organization="sentence-transformers-testing")
        assert mock_upload_folder_kwargs["repo_id"] == "sentence-transformers-testing/stsb-bert-tiny-safetensors"
        assert url == "https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors/commit/123456"
        assert len(caplog.record_tuples) == 2
        assert (
            caplog.record_tuples[0][2]
            == "The `save_to_hub` method is deprecated and will be removed in a future version of SentenceTransformers. Please use `push_to_hub` instead for future model uploads."
        )
        assert (
            caplog.record_tuples[1][2]
            == 'Providing an `organization` to `save_to_hub` is deprecated, please use `repo_id="sentence-transformers-testing/stsb-bert-tiny-safetensors"` instead.'
        )
    mock_upload_folder_kwargs.clear()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        url = model.save_to_hub(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors", local_model_path="my_fake_local_model_path"
        )
        assert mock_upload_folder_kwargs["repo_id"] == "sentence-transformers-testing/stsb-bert-tiny-safetensors"
        assert mock_upload_folder_kwargs["folder_path"] == "my_fake_local_model_path"
        assert url == "https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors/commit/123456"
        assert len(caplog.record_tuples) == 1
        assert (
            caplog.record_tuples[0][2]
            == "The `save_to_hub` method is deprecated and will be removed in a future version of SentenceTransformers. Please use `push_to_hub` instead for future model uploads."
        )
    mock_upload_folder_kwargs.clear()

    # Incorrect usage: Using deprecated "repo_name" positional argument
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        url = model.save_to_hub(repo_name="sentence-transformers-testing/stsb-bert-tiny-safetensors")
        assert mock_upload_folder_kwargs["repo_id"] == "sentence-transformers-testing/stsb-bert-tiny-safetensors"
        assert url == "https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors/commit/123456"
        assert len(caplog.record_tuples) == 2
        assert (
            caplog.record_tuples[0][2]
            == "Providing a `repo_name` keyword argument to `save_to_hub` is deprecated, please use `repo_id` instead."
        )
        assert (
            caplog.record_tuples[1][2]
            == "The `save_to_hub` method is deprecated and will be removed in a future version of SentenceTransformers. Please use `push_to_hub` instead for future model uploads."
        )
    mock_upload_folder_kwargs.clear()

    # Incorrect usage: Use positional arguments from before "token" was introduced
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        url = model.save_to_hub(
            "stsb-bert-tiny-safetensors",  # repo_name
            "sentence-transformers-testing",  # organization
            True,  # private
            commit_message="Adding new awesome Model!",
            exist_ok=True,
        )
        assert mock_upload_folder_kwargs["repo_id"] == "sentence-transformers-testing/stsb-bert-tiny-safetensors"
        assert mock_upload_folder_kwargs["commit_message"] == "Adding new awesome Model!"
        assert url == "https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors/commit/123456"
        assert len(caplog.record_tuples) == 2
        assert (
            caplog.record_tuples[0][2]
            == "The `save_to_hub` method is deprecated and will be removed in a future version of SentenceTransformers. Please use `push_to_hub` instead for future model uploads."
        )
        assert (
            caplog.record_tuples[1][2]
            == 'Providing an `organization` to `save_to_hub` is deprecated, please use `repo_id="sentence-transformers-testing/stsb-bert-tiny-safetensors"` instead.'
        )
    mock_upload_folder_kwargs.clear()

    url = model.push_to_hub("sentence-transformers-testing/stsb-bert-tiny-safetensors", revision="revision_test")
    assert mock_upload_folder_kwargs["repo_id"] == "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    assert mock_upload_folder_kwargs["revision"] == "revision_test"
    assert url == "https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors/commit/678901"
    mock_upload_folder_kwargs.clear()


@pytest.mark.parametrize("safe_serialization", [True, False, None])
def test_safe_serialization(safe_serialization: bool) -> None:
    with SafeTemporaryDirectory() as cache_folder:
        model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        if safe_serialization:
            model.save(cache_folder, safe_serialization=safe_serialization)
            model_files = list(Path(cache_folder).glob("**/model.safetensors"))
            assert 1 == len(model_files)
        elif safe_serialization is None:
            model.save(cache_folder)
            model_files = list(Path(cache_folder).glob("**/model.safetensors"))
            assert 1 == len(model_files)
        else:
            model.save(cache_folder, safe_serialization=safe_serialization)
            model_files = list(Path(cache_folder).glob("**/pytorch_model.bin"))
            assert 1 == len(model_files)


def test_load_with_revision() -> None:
    main_model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors", revision="main")
    latest_model = SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors", revision="f3cb857cba53019a20df283396bcca179cf051a4"
    )
    older_model = SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors", revision="ba33022fdf0b0fc2643263f0726f44d0a07d0e24"
    )

    test_sentence = ["Hello there!"]
    main_embeddings = main_model.encode(test_sentence, convert_to_tensor=True)
    assert torch.equal(main_embeddings, latest_model.encode(test_sentence, convert_to_tensor=True))
    assert not torch.equal(main_embeddings, older_model.encode(test_sentence, convert_to_tensor=True))


def test_load_local_without_normalize_directory() -> None:
    tiny_model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    tiny_model.add_module("Normalize", Normalize())
    with SafeTemporaryDirectory() as tmp_folder:
        model_path = Path(tmp_folder) / "tiny_model_local"
        tiny_model.save(str(model_path))

        assert (model_path / "2_Normalize").exists()
        os.rmdir(model_path / "2_Normalize")
        assert not (model_path / "2_Normalize").exists()

        # This fails in v2.3.0
        fresh_tiny_model = SentenceTransformer(str(model_path))
        assert isinstance(fresh_tiny_model, SentenceTransformer)


def test_prompts(caplog: pytest.LogCaptureFixture) -> None:
    model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    assert model.prompts == {}
    assert model.default_prompt_name is None
    texts = ["How to bake a chocolate cake", "Symptoms of the flu"]
    no_prompt_embedding = model.encode(texts)
    prompt_embedding = model.encode([f"query: {text}" for text in texts])
    assert not np.array_equal(no_prompt_embedding, prompt_embedding)

    for query in ["query: ", "query:", "query:   "]:
        # Test prompt="... {}"
        model.prompts = {}
        assert np.array_equal(model.encode(texts, prompt=query), prompt_embedding)

        # Test prompt_name="..."
        model.prompts = {"query": query}
        assert np.array_equal(model.encode(texts, prompt_name="query"), prompt_embedding)

        caplog.clear()
        # Test prompt_name="..." & prompt="..."
        with caplog.at_level(logging.WARNING):
            assert np.array_equal(model.encode(texts, prompt=query, prompt_name="query"), prompt_embedding)
            assert len(caplog.record_tuples) == 1
            assert (
                caplog.record_tuples[0][2]
                == "Encode with either a `prompt`, a `prompt_name`, or neither, but not both. "
                "Ignoring the `prompt_name` in favor of `prompt`."
            )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Prompt name 'invalid_prompt_name' not found in the configured prompts dictionary with keys ['query']."
            ),
        ):
            model.encode(texts, prompt_name="invalid_prompt_name")


def test_save_load_prompts() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Default prompt name 'invalid_prompt_name' not found in the configured prompts dictionary with keys ['query']."
        ),
    ):
        model = SentenceTransformer(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors",
            prompts={"query": "query: "},
            default_prompt_name="invalid_prompt_name",
        )

    model = SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        prompts={"query": "query: "},
        default_prompt_name="query",
    )
    assert model.prompts == {"query": "query: "}
    assert model.default_prompt_name == "query"

    with SafeTemporaryDirectory() as tmp_folder:
        model_path = Path(tmp_folder) / "tiny_model_local"
        model.save(str(model_path))
        config_path = model_path / "config_sentence_transformers.json"
        assert config_path.exists()
        with open(config_path, encoding="utf8") as f:
            saved_config = json.load(f)
        assert saved_config["prompts"] == {"query": "query: "}
        assert saved_config["default_prompt_name"] == "query"

        fresh_model = SentenceTransformer(str(model_path))
        assert fresh_model.prompts == {"query": "query: "}
        assert fresh_model.default_prompt_name == "query"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA must be available to test float16 support.")
def test_load_with_torch_dtype() -> None:
    model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")

    assert model.encode(["Hello there!"], convert_to_tensor=True).dtype == torch.float32

    with SafeTemporaryDirectory() as tmp_folder:
        fp16_model_dir = Path(tmp_folder) / "fp16_model"
        model.half()
        model.save(str(fp16_model_dir))
        del model

        fp16_model = SentenceTransformer(
            str(fp16_model_dir),
            model_kwargs={"torch_dtype": "auto"},
        )
        assert fp16_model.encode(["Hello there!"], convert_to_tensor=True).dtype == torch.float16


def test_load_with_model_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    transformer_kwargs = {}
    original_transformer_init = Transformer.__init__

    def transformers_init(*args, **kwargs):
        nonlocal transformer_kwargs
        nonlocal original_transformer_init
        transformer_kwargs = kwargs
        return original_transformer_init(*args, **kwargs)

    monkeypatch.setattr(Transformer, "__init__", transformers_init)

    SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        model_kwargs={"attn_implementation": "eager", "low_cpu_mem_usage": False},
    )

    assert "low_cpu_mem_usage" in transformer_kwargs["model_args"]
    assert transformer_kwargs["model_args"]["low_cpu_mem_usage"] is False
    assert "attn_implementation" in transformer_kwargs["model_args"]
    assert transformer_kwargs["model_args"]["attn_implementation"] == "eager"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA must be available to test float16 support.")
def test_encode_fp16() -> None:
    tiny_model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    tiny_model.half()
    embeddings = tiny_model.encode(["Hello there!"], convert_to_tensor=True)
    assert embeddings.dtype == torch.float16


@pytest.mark.parametrize("convert_to_tensor", [True, False])
@pytest.mark.parametrize("convert_to_numpy", [True, False])
@pytest.mark.parametrize(
    ("precision", "expected_torch_dtype", "expected_numpy_dtype"),
    [
        (None, torch.float32, np.float32),
        ("float32", torch.float32, np.float32),
        ("int8", torch.int8, np.int8),
        ("uint8", torch.uint8, np.uint8),
        ("binary", torch.int8, np.int8),
        ("ubinary", torch.uint8, np.uint8),
    ],
)
def test_encode_quantization(
    stsb_bert_tiny_model_reused: SentenceTransformer,
    convert_to_tensor: bool,
    convert_to_numpy: bool,
    precision: str,
    expected_torch_dtype,
    expected_numpy_dtype,
) -> None:
    tiny_model = stsb_bert_tiny_model_reused
    embeddings = tiny_model.encode(
        ["One sentence", "Another sentence"],
        convert_to_tensor=convert_to_tensor,
        convert_to_numpy=convert_to_numpy,
        precision=precision,
    )
    if convert_to_tensor:
        assert embeddings[0].dtype == expected_torch_dtype
        assert isinstance(embeddings, torch.Tensor)
    elif convert_to_numpy:
        assert embeddings[0].dtype == expected_numpy_dtype
        assert isinstance(embeddings, np.ndarray)
    else:
        assert embeddings[0].dtype == expected_torch_dtype
        assert isinstance(embeddings, list)


@pytest.mark.parametrize("sentences", ("Single sentence", ["One sentence", "Another sentence"]))
@pytest.mark.parametrize("convert_to_tensor", [True, False])
@pytest.mark.parametrize("convert_to_numpy", [True, False])
@pytest.mark.parametrize("normalize_embeddings", [True, False])
@pytest.mark.parametrize("output_value", ["sentence_embedding", None])
def test_encode_truncate(
    sentences: str | list[str],
    convert_to_tensor: bool,
    convert_to_numpy: bool,
    normalize_embeddings: bool,
    output_value: Literal["sentence_embedding"] | None,
) -> None:
    model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    embeddings_full_unnormalized: torch.Tensor = model.encode(
        sentences, convert_to_numpy=False, convert_to_tensor=True
    )  # These are raw embeddings which serve as the reference to test against

    def test(model: SentenceTransformer, expected_dim: int):
        outputs = model.encode(
            sentences,
            output_value=output_value,
            convert_to_tensor=convert_to_tensor,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
        )

        # Extract the sentence embeddings out of outputs
        if output_value is None:
            # We get the whole plate
            if not isinstance(outputs, List):
                embeddings = outputs["sentence_embedding"]
            else:
                outputs = cast(List[Dict[str, torch.Tensor]], outputs)
                embeddings = [out_features["sentence_embedding"] for out_features in outputs]
        else:
            embeddings = outputs

        # Test shape
        if isinstance(embeddings, list):  # list of tensors
            embeddings_shape = (len(embeddings), embeddings[0].shape[-1])
        else:
            embeddings_shape = embeddings.shape
        expected_shape = (expected_dim,) if isinstance(sentences, str) else (len(sentences), expected_dim)
        assert embeddings_shape == expected_shape
        assert model.get_sentence_embedding_dimension() == expected_dim

        # Convert embeddings to a torch Tensor for ease of testing
        if isinstance(embeddings, list):
            embeddings = torch.stack(embeddings)
        elif isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).to(embeddings_full_unnormalized.device)
            # On a non-cpu device, the device of torch.from_numpy(embeddings) is always CPU

        # Test content
        if normalize_embeddings:
            if output_value is None:
                # Currently, normalization is not performed; it's the raw output of the forward pass
                pass
            else:
                normalize = partial(torch.nn.functional.normalize, p=2, dim=-1)
                assert torch.allclose(
                    embeddings,
                    normalize(util.truncate_embeddings(embeddings_full_unnormalized, expected_dim)),
                )
        else:
            assert torch.allclose(embeddings, util.truncate_embeddings(embeddings_full_unnormalized, expected_dim))

    # Test init w/o setting truncate_dim (it's None)
    original_output_dim: int = model.get_sentence_embedding_dimension()
    test(model, expected_dim=original_output_dim)

    # Test init w/ a set truncate_dim
    truncate_dim = int(original_output_dim / 4)
    model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors", truncate_dim=truncate_dim)
    test(model, expected_dim=truncate_dim)

    # Test setting the attribute after init to a greater dimension
    new_truncate_dim = 2 * truncate_dim
    model.truncate_dim = new_truncate_dim
    test(model, expected_dim=new_truncate_dim)

    # Test context manager
    final_truncate_dim = int(original_output_dim / 8)
    with model.truncate_sentence_embeddings(final_truncate_dim):
        test(model, expected_dim=final_truncate_dim)
    test(model, expected_dim=new_truncate_dim)  # b/c we've exited the context

    # Test w/ an ouptut_dim that's larger than the original_output_dim. No truncation ends up happening
    model.truncate_dim = 2 * original_output_dim
    test(model, expected_dim=original_output_dim)


@pytest.mark.parametrize("similarity_fn_name", SimilarityFunction.possible_values())
def test_similarity_score(stsb_bert_tiny_model_reused: SentenceTransformer, similarity_fn_name: str) -> None:
    model = stsb_bert_tiny_model_reused
    model.similarity_fn_name = similarity_fn_name
    sentences = [
        "The weather is so nice!",
        "It's so sunny outside.",
        "He's driving to the movie theater.",
        "She's going to the cinema.",
    ]
    embeddings = model.encode(sentences, normalize_embeddings=True)
    scores = model.similarity(embeddings, embeddings)
    assert scores.shape == (len(sentences), len(sentences))
    if similarity_fn_name in ("cosine", "dot"):
        expected = np.ones(4, dtype=float)
    else:
        expected = np.zeros(4, dtype=float)
    np.testing.assert_almost_equal(np.diag(scores), expected, decimal=4)
    assert scores[1][0] > scores[2][0]
    assert scores[1][0] > scores[3][0]
    assert scores[2][3] > scores[2][0]
    assert scores[2][3] > scores[2][1]

    pairwise_scores = model.similarity_pairwise(embeddings[::2], embeddings[1::2])
    assert pairwise_scores.shape == (len(sentences) // 2,)
    if similarity_fn_name in ("cosine", "dot"):
        assert (pairwise_scores > 0.5).all()


def test_similarity_score_save(stsb_bert_tiny_model: SentenceTransformer) -> None:
    model = stsb_bert_tiny_model
    embeddings = model.encode(["Sentence 1", "Sentence 2"])
    assert model.similarity_fn_name is None
    cosine_scores = model.similarity(embeddings, embeddings)
    # Using 'similarity' methods sets the default similarity function to 'cosine'
    assert model.similarity_fn_name == "cosine"

    model.similarity_fn_name = "euclidean"
    with SafeTemporaryDirectory() as tmp_folder:
        model.save(tmp_folder)
        loaded_model = SentenceTransformer(tmp_folder)
    assert loaded_model.similarity_fn_name == "euclidean"
    dot_scores = model.similarity(embeddings, embeddings)
    assert np.not_equal(cosine_scores, dot_scores).all()


def test_model_card_save_update_model_id(stsb_bert_tiny_model: SentenceTransformer) -> None:
    model = stsb_bert_tiny_model
    # Removing the saved model card will cause a fresh one to be generated when we save
    model._model_card_text = ""
    with SafeTemporaryDirectory() as tmp_folder:
        model.save(tmp_folder)
        with open(Path(tmp_folder) / "README.md", encoding="utf8") as f:
            model_card_text = f.read()
            assert 'model = SentenceTransformer("sentence_transformers_model_id"' in model_card_text

        # When we reload this saved model and then re-save it, we want to override the 'sentence_transformers_model_id'
        # if we have it set
        loaded_model = SentenceTransformer(tmp_folder)

    with SafeTemporaryDirectory() as tmp_folder:
        loaded_model.save(tmp_folder, model_name="test_user/test_model")

        with open(Path(tmp_folder) / "README.md", encoding="utf8") as f:
            model_card_text = f.read()
            assert 'model = SentenceTransformer("test_user/test_model"' in model_card_text


def test_override_config_versions(stsb_bert_tiny_model: SentenceTransformer) -> None:
    model = stsb_bert_tiny_model

    assert model._model_config["__version__"]["sentence_transformers"] == "2.2.2"
    with SafeTemporaryDirectory() as tmp_folder:
        model.save(tmp_folder)
        loaded_model = SentenceTransformer(tmp_folder)
    # Verify that the version has now been updated when saving the model again
    assert loaded_model._model_config["__version__"]["sentence_transformers"] != "2.2.2"


@pytest.mark.parametrize(
    "modules",
    [
        [
            Transformer("sentence-transformers-testing/stsb-bert-tiny-safetensors"),
            Pooling(128, "mean"),
            Dense(128, 128),
        ],
        [Transformer("sentence-transformers-testing/stsb-bert-tiny-safetensors"), CNN(128, 128), Pooling(128, "mean")],
        [
            Transformer("sentence-transformers-testing/stsb-bert-tiny-safetensors"),
            Pooling(128, "mean"),
            LayerNorm(128),
        ],
        [
            SentenceTransformer("sentence-transformers/average_word_embeddings_levy_dependency")[0],
            LSTM(300, 128),
            Pooling(128, "mean"),
        ],
        [
            Transformer("sentence-transformers-testing/stsb-bert-tiny-safetensors"),
            WeightedLayerPooling(128, num_hidden_layers=2, layer_start=1),
            Pooling(128, "mean"),
        ],
        SentenceTransformer("sentence-transformers/average_word_embeddings_levy_dependency"),
    ],
)
def test_safetensors(modules: list[nn.Module] | SentenceTransformer) -> None:
    if isinstance(modules, SentenceTransformer):
        model = modules
    else:
        # output_hidden_states must be True for WeightedLayerPooling
        if isinstance(modules[1], WeightedLayerPooling):
            modules[0].auto_model.config.output_hidden_states = True
        model = SentenceTransformer(modules=modules)
    original_embedding = model.encode("Hello, World!")

    with SafeTemporaryDirectory() as tmp_folder:
        model.save(tmp_folder)
        # Ensure that we only have the safetensors file and no pytorch_model.bin
        assert list(Path(tmp_folder).rglob("**/model.safetensors"))
        assert not list(Path(tmp_folder).rglob("**/pytorch_model.bin"))

        # Ensure that we can load the model again and get the same embeddings
        loaded_model = SentenceTransformer(tmp_folder)
        loaded_embedding = loaded_model.encode("Hello, World!")
        assert np.allclose(original_embedding, loaded_embedding)


def test_empty_encode(stsb_bert_tiny_model: SentenceTransformer) -> None:
    model = stsb_bert_tiny_model
    embeddings = model.encode([])
    assert embeddings.shape == (0,)
