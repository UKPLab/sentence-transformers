"""
Tests general behaviour of the SentenceTransformer class
"""

from functools import partial
import json
import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Dict, List, Literal, Optional, Union, cast

import numpy as np
import pytest

from huggingface_hub import HfApi, RepoUrl, GitRefs, GitRefInfo
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Normalize, Transformer, Pooling
from sentence_transformers import util


def test_load_with_safetensors() -> None:
    with tempfile.TemporaryDirectory() as cache_folder:
        safetensors_model = SentenceTransformer(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors",
            cache_folder=cache_folder,
        )

        # Only the safetensors file must be loaded
        pytorch_files = list(Path(cache_folder).glob("**/pytorch_model.bin"))
        assert 0 == len(pytorch_files), "PyTorch model file must not be downloaded."
        safetensors_files = list(Path(cache_folder).glob("**/model.safetensors"))
        assert 1 == len(safetensors_files), "Safetensors model file must be downloaded."

    with tempfile.TemporaryDirectory() as cache_folder:
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


def test_push_to_hub(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def mock_create_repo(self, repo_id, **kwargs):
        return RepoUrl(f"https://huggingface.co/{repo_id}")

    mock_upload_folder_kwargs = {}

    def mock_upload_folder(self, **kwargs):
        nonlocal mock_upload_folder_kwargs
        mock_upload_folder_kwargs = kwargs

    def mock_list_repo_refs(self, repo_id=None, **kwargs):
        try:
            git_ref_info = GitRefInfo(name="main", ref="refs/heads/main", target_commit="123456")
        except TypeError:
            git_ref_info = GitRefInfo(dict(name="main", ref="refs/heads/main", targetCommit="123456"))
        # workaround for https://github.com/huggingface/huggingface_hub/issues/1956
        git_ref_kwargs = {"branches": [git_ref_info], "converts": [], "tags": [], "pull_requests": None}
        try:
            return GitRefs(**git_ref_kwargs)
        except TypeError:
            git_ref_kwargs.pop("pull_requests")
            return GitRefs(**git_ref_kwargs)

    monkeypatch.setattr(HfApi, "create_repo", mock_create_repo)
    monkeypatch.setattr(HfApi, "upload_folder", mock_upload_folder)
    monkeypatch.setattr(HfApi, "list_repo_refs", mock_list_repo_refs)

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


@pytest.mark.parametrize("safe_serialization", [True, False, None])
def test_safe_serialization(safe_serialization: bool) -> None:
    with tempfile.TemporaryDirectory() as cache_folder:
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
    with tempfile.TemporaryDirectory() as tmp_folder:
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

    with tempfile.TemporaryDirectory() as tmp_folder:
        model_path = Path(tmp_folder) / "tiny_model_local"
        model.save(str(model_path))
        config_path = model_path / "config_sentence_transformers.json"
        assert config_path.exists()
        with open(config_path, "r", encoding="utf8") as f:
            saved_config = json.load(f)
        assert saved_config["prompts"] == {"query": "query: "}
        assert saved_config["default_prompt_name"] == "query"

        fresh_model = SentenceTransformer(str(model_path))
        assert fresh_model.prompts == {"query": "query: "}
        assert fresh_model.default_prompt_name == "query"


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
    sentences: Union[str, List[str]],
    convert_to_tensor: bool,
    convert_to_numpy: bool,
    normalize_embeddings: bool,
    output_value: Optional[Literal["sentence_embedding"]],
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
                # TODO: can overload model.encode if ppl want type checker compatibility
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
