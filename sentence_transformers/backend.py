from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

import huggingface_hub

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

    try:
        from optimum.onnxruntime.configuration import OptimizationConfig, QuantizationConfig
        from optimum.intel import OVQuantizationConfig
    except ImportError:
        pass


def export_optimized_onnx_model(
    model: SentenceTransformer,
    optimization_config: OptimizationConfig | Literal["O1", "O2", "O3", "O4"],
    model_name_or_path: str,
    push_to_hub: bool = False,
    create_pr: bool = False,
    file_suffix: str | None = None,
) -> None:
    """
    Export an optimized ONNX model from a SentenceTransformer model.

    The O1-O4 optimization levels are defined by Optimum and are documented here:
    https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/optimization

    The optimization levels are:

    - O1: basic general optimizations.
    - O2: basic and extended general optimizations, transformers-specific fusions.
    - O3: same as O2 with GELU approximation.
    - O4: same as O3 with mixed precision (fp16, GPU-only)

    See https://sbert.net/docs/sentence_transformer/usage/efficiency.html for more information & benchmarks.

    Args:
        model (SentenceTransformer): The SentenceTransformer model to be optimized. Must be loaded with `backend="onnx"`.
        optimization_config (OptimizationConfig | Literal["O1", "O2", "O3", "O4"]): The optimization configuration or level.
        model_name_or_path (str): The path or Hugging Face Hub repository name where the optimized model will be saved.
        push_to_hub (bool, optional): Whether to push the optimized model to the Hugging Face Hub. Defaults to False.
        create_pr (bool, optional): Whether to create a pull request when pushing to the Hugging Face Hub. Defaults to False.
        file_suffix (str | None, optional): The suffix to add to the optimized model file name. Defaults to None.

    Raises:
        ImportError: If the required packages `optimum` and `onnxruntime` are not installed.
        ValueError: If the provided model is not a valid SentenceTransformer model loaded with `backend="onnx"`.
        ValueError: If the provided optimization_config is not valid.

    Returns:
        None
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models.Transformer import Transformer

    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTOptimizer
        from optimum.onnxruntime.configuration import AutoOptimizationConfig
    except ImportError:
        raise ImportError(
            "Please install Optimum and ONNX Runtime to use this function. "
            "You can install them with pip: `pip install optimum[onnxruntime]` "
            "or `pip install optimum[onnxruntime-gpu]`"
        )

    if (
        not isinstance(model, SentenceTransformer)
        or not len(model)
        or not isinstance(model[0], Transformer)
        or not isinstance(model[0].auto_model, ORTModelForFeatureExtraction)
    ):
        raise ValueError(
            'The model must be a Transformer-based SentenceTransformer model loaded with `backend="onnx"`.'
        )

    ort_model: ORTModelForFeatureExtraction = model[0].auto_model
    optimizer = ORTOptimizer.from_pretrained(ort_model)

    if isinstance(optimization_config, str):
        if optimization_config not in AutoOptimizationConfig._LEVELS:
            raise ValueError(
                "optimization_config must be an OptimizationConfig instance or one of 'O1', 'O2', 'O3', 'O4'."
            )

        file_suffix = file_suffix or optimization_config
        optimization_config = getattr(AutoOptimizationConfig, optimization_config)()

    if file_suffix is None:
        file_suffix = "optimized"

    save_or_push_to_hub_model(
        export_function=lambda save_dir: optimizer.optimize(optimization_config, save_dir, file_suffix=file_suffix),
        export_function_name="export_optimized_onnx_model",
        config=optimization_config,
        model_name_or_path=model_name_or_path,
        push_to_hub=push_to_hub,
        create_pr=create_pr,
        file_suffix=file_suffix,
        backend="onnx",
    )


def export_dynamic_quantized_onnx_model(
    model: SentenceTransformer,
    quantization_config: QuantizationConfig | Literal["arm64", "avx2", "avx512", "avx512_vnni"],
    model_name_or_path: str,
    push_to_hub: bool = False,
    create_pr: bool = False,
    file_suffix: str | None = None,
) -> None:
    """
    Export a quantized ONNX model from a SentenceTransformer model.

    This function applies dynamic quantization, i.e. without a calibration dataset.
    Each of the default quantization configurations quantize the model to int8, allowing
    for faster inference on CPUs, but are likely slower on GPUs.

    See https://sbert.net/docs/sentence_transformer/usage/efficiency.html for more information & benchmarks.

    Args:
        model (SentenceTransformer): The SentenceTransformer model to be quantized. Must be loaded with `backend="onnx"`.
        quantization_config (QuantizationConfig): The quantization configuration.
        model_name_or_path (str): The path or Hugging Face Hub repository name where the quantized model will be saved.
        push_to_hub (bool, optional): Whether to push the quantized model to the Hugging Face Hub. Defaults to False.
        create_pr (bool, optional): Whether to create a pull request when pushing to the Hugging Face Hub. Defaults to False.
        file_suffix (str | None, optional): The suffix to add to the quantized model file name. Defaults to None.

    Raises:
        ImportError: If the required packages `optimum` and `onnxruntime` are not installed.
        ValueError: If the provided model is not a valid SentenceTransformer model loaded with `backend="onnx"`.
        ValueError: If the provided quantization_config is not valid.

    Returns:
        None
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models.Transformer import Transformer

    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
    except ImportError:
        raise ImportError(
            "Please install Optimum and ONNX Runtime to use this function. "
            "You can install them with pip: `pip install optimum[onnxruntime]` "
            "or `pip install optimum[onnxruntime-gpu]`"
        )

    if (
        not isinstance(model, SentenceTransformer)
        or not len(model)
        or not isinstance(model[0], Transformer)
        or not isinstance(model[0].auto_model, ORTModelForFeatureExtraction)
    ):
        raise ValueError(
            'The model must be a Transformer-based SentenceTransformer model loaded with `backend="onnx"`.'
        )

    ort_model: ORTModelForFeatureExtraction = model[0].auto_model
    quantizer = ORTQuantizer.from_pretrained(ort_model)

    if isinstance(quantization_config, str):
        if quantization_config not in ["arm64", "avx2", "avx512", "avx512_vnni"]:
            raise ValueError(
                "quantization_config must be an QuantizationConfig instance or one of 'arm64', 'avx2', 'avx512', or 'avx512_vnni'."
            )

        quantization_config_name = quantization_config[:]
        quantization_config = getattr(AutoQuantizationConfig, quantization_config)(is_static=False)
        file_suffix = file_suffix or f"{quantization_config.weights_dtype.name.lower()}_{quantization_config_name}"

    if file_suffix is None:
        file_suffix = f"{quantization_config.weights_dtype.name.lower()}_quantized"

    save_or_push_to_hub_model(
        export_function=lambda save_dir: quantizer.quantize(quantization_config, save_dir, file_suffix=file_suffix),
        export_function_name="export_dynamic_quantized_onnx_model",
        config=quantization_config,
        model_name_or_path=model_name_or_path,
        push_to_hub=push_to_hub,
        create_pr=create_pr,
        file_suffix=file_suffix,
        backend="onnx",
    )


def export_static_quantized_openvino_model(
    model: SentenceTransformer,
    quantization_config: OVQuantizationConfig,
    model_name_or_path: str,
    push_to_hub: bool = False,
    create_pr: bool = False,
    file_suffix: str = "qint8_quantized",
) -> None:
    """
    Export a quantized OpenVINO model from a SentenceTransformer model.

    This function applies Post-Training Static Quantization (PTQ) using a calibration dataset, which calibrates
    quantization constants without requiring model retraining. Each default quantization configuration converts
    the model to int8 precision, enabling faster inference while maintaining accuracy.

    See https://sbert.net/docs/sentence_transformer/usage/efficiency.html for more information & benchmarks.

    Args:
        model (SentenceTransformer): The SentenceTransformer model to be quantized. Must be loaded with `backend="openvino"`.
        quantization_config (OVQuantizationConfig): The quantization configuration.
        model_name_or_path (str): The path or Hugging Face Hub repository name where the quantized model will be saved.
        push_to_hub (bool, optional): Whether to push the quantized model to the Hugging Face Hub. Defaults to False.
        create_pr (bool, optional): Whether to create a pull request when pushing to the Hugging Face Hub. Defaults to False.
        file_suffix (str, optional): The suffix to add to the quantized model file name. Defaults to `qint8_quantized`.

    Raises:
        ImportError: If the required packages `optimum` and `openvino` are not installed.
        ValueError: If the provided model is not a valid SentenceTransformer model loaded with `backend="openvino"`.
        ValueError: If the provided quantization_config is not valid.

    Returns:
        None
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models.Transformer import Transformer

    try:
        from optimum.intel import OVModelForFeatureExtraction, OVQuantizer, OVConfig
    except ImportError:
        raise ImportError(
            "Please install Optimum and OpenVINO to use this function. "
            "You can install them with pip: `pip install optimum[openvino]`"
        )

    if (
        not isinstance(model, SentenceTransformer)
        or not len(model)
        or not isinstance(model[0], Transformer)
        or not isinstance(model[0].auto_model, OVModelForFeatureExtraction)
    ):
        raise ValueError(
            'The model must be a Transformer-based SentenceTransformer model loaded with `backend="openvino"`.'
        )

    ov_model: OVModelForFeatureExtraction = model[0].auto_model
    ov_config = OVConfig(quantization_config=quantization_config)
    quantizer = OVQuantizer.from_pretrained(ov_model)

    def preprocess_function(examples):
        return model.tokenizer(examples["sentence"], padding="max_length", max_length=384, truncation=True)

    calibration_dataset = quantizer.get_calibration_dataset(
        dataset_name="glue",
        dataset_config_name="sst2",
        preprocess_function=preprocess_function,
        num_samples=300,
        dataset_split="train",
    )

    save_or_push_to_hub_model(
        export_function=lambda save_dir: quantizer.quantize(calibration_dataset, save_directory=save_dir, ov_config=ov_config),
        export_function_name="export_static_quantized_openvino_model",
        config=quantization_config,
        model_name_or_path=model_name_or_path,
        push_to_hub=push_to_hub,
        create_pr=create_pr,
        file_suffix=file_suffix,
        backend="openvino",
    )


def save_or_push_to_hub_model(
    export_function: Callable,
    export_function_name: str,
    config,
    model_name_or_path: str,
    push_to_hub: bool = False,
    create_pr: bool = False,
    file_suffix: str | None = None,
    backend: str = "onnx",
):
    if backend == "onnx":
        file_name = f"model_{file_suffix}.onnx"
    elif backend == "openvino":
        file_name = f"openvino_model.xml"
        destination_file_name = Path(f"openvino_model_{file_suffix}.xml")

    if push_to_hub:
        with tempfile.TemporaryDirectory() as save_dir:
            export_function(save_dir)
            if backend == "onnx":
                source = (Path(save_dir) / file_name).as_posix()
                destination = Path(backend) / file_name
            elif backend == "openvino":
                source = (Path(save_dir) / backend / file_name).as_posix()
                destination = Path(backend) / destination_file_name
            else:
                raise NotImplementedError(f"Unsupported backend type: {backend}")

            commit_description = ""
            if create_pr:
                opt_config_string = repr(config).replace("(", "(\n\t").replace(", ", ",\n\t").replace(")", "\n)")
                commit_description = f"""\
Hello!

*This pull request has been automatically generated from the [`{export_function_name}`](https://sbert.net/docs/package_reference/util.html#sentence_transformers.backend.{export_function_name}) function from the Sentence Transformers library.*

## Config
```python
{opt_config_string}
```

## Tip:
Consider testing this pull request before merging by loading the model from this PR with the `revision` argument:
```python
from sentence_transformers import SentenceTransformer

# TODO: Fill in the PR number
pr_number = 2
model = SentenceTransformer(
    "{model_name_or_path}",
    revision=f"refs/pr/{{pr_number}}",
    backend="{backend}",
    model_kwargs={{"file_name": "{destination}"}},
)

# Verify that everything works as expected
embeddings = model.encode(["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."])
print(embeddings.shape)

similarities = model.similarity(embeddings, embeddings)
print(similarities)
```
"""

            huggingface_hub.upload_file(
                path_or_fileobj=source,
                path_in_repo=destination.as_posix(),
                repo_id=model_name_or_path,
                repo_type="model",
                commit_message=f"Add exported {backend} model {destination.name!r}",
                commit_description=commit_description,
                create_pr=create_pr,
            )

    else:
        with tempfile.TemporaryDirectory() as save_dir:
            export_function(save_dir)

            dst_dir = os.path.join(model_name_or_path, backend)
            # Create destination if it does not exist
            os.makedirs(dst_dir, exist_ok=True)

            if backend == "openvino":
                source = Path(save_dir) / backend / file_name
                bin_file = source.with_suffix(".bin")
                xml_destination = os.path.join(dst_dir, destination_file_name)
                bin_destination = os.path.join(dst_dir, destination_file_name.with_suffix(".bin"))
                shutil.copy(source, xml_destination)
                shutil.copy(bin_file, bin_destination)
            else:
                source = os.path.join(save_dir, file_name)
                destination = os.path.join(dst_dir, file_name)
                shutil.copy(source, destination)
