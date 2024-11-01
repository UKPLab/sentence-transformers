from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

import huggingface_hub

from sentence_transformers.util import disable_datasets_caching, is_datasets_available

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

    try:
        from optimum.intel import OVQuantizationConfig
    except ImportError:
        pass
    try:
        from optimum.onnxruntime.configuration import OptimizationConfig, QuantizationConfig
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
    quantization_config: OVQuantizationConfig | dict | None,
    model_name_or_path: str,
    dataset_name: str | None = None,
    dataset_config_name: str | None = None,
    dataset_split: str | None = None,
    column_name: str | None = None,
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
        quantization_config (OVQuantizationConfig | dict | None): The quantization configuration. If None, default values are used.
        model_name_or_path (str): The path or Hugging Face Hub repository name where the quantized model will be saved.
        dataset_name(str, optional): The name of the dataset to load for calibration.
            If not specified, the `sst2` subset of the `glue` dataset will be used by default.
        dataset_config_name (str, optional): The specific configuration of the dataset to load.
        dataset_split (str, optional): The split of the dataset to load (e.g., 'train', 'test'). Defaults to None.
        column_name (str, optional): The column name in the dataset to use for calibration. Defaults to None.
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
        from optimum.intel import OVConfig, OVModelForFeatureExtraction, OVQuantizationConfig, OVQuantizer
    except ImportError:
        raise ImportError(
            "Please install datasets, optimum-intel and openvino to use this function. "
            "You can install them with pip: `pip install datasets optimum[openvino]`"
        )
    if not is_datasets_available():
        raise ImportError(
            "Please install datasets to use this function. You can install it with pip: `pip install datasets`"
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

    if quantization_config is None:
        quantization_config = OVQuantizationConfig()

    ov_model: OVModelForFeatureExtraction = model[0].auto_model
    ov_config = OVConfig(quantization_config=quantization_config)
    quantizer = OVQuantizer.from_pretrained(ov_model)

    if any(param is not None for param in [dataset_name, dataset_config_name, dataset_split, column_name]) and not all(
        param is not None for param in [dataset_name, dataset_config_name, dataset_split, column_name]
    ):
        raise ValueError(
            "Either specify all of `dataset_name`, `dataset_config_name`, `dataset_split`, and `column_name`, or leave them all unspecified."
        )

    def preprocess_function(examples):
        return model.tokenizer(examples, padding="max_length", max_length=384, truncation=True)

    dataset_name = dataset_name if dataset_name is not None else "glue"
    dataset_config_name = dataset_config_name if dataset_config_name is not None else "sst2"
    dataset_split = dataset_split if dataset_split is not None else "train"
    column_name = column_name if column_name is not None else "sentence"
    with disable_datasets_caching():
        calibration_dataset = quantizer.get_calibration_dataset(
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            preprocess_function=lambda examples: preprocess_function(examples[column_name]),
            num_samples=quantization_config.num_samples if quantization_config is not None else 300,
            dataset_split=dataset_split,
        )

    save_or_push_to_hub_model(
        export_function=lambda save_dir: quantizer.quantize(
            calibration_dataset, save_directory=save_dir, ov_config=ov_config
        ),
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
        file_name = f"openvino_model_{file_suffix}.xml"

    with tempfile.TemporaryDirectory() as save_dir:
        export_function(save_dir)

        # OpenVINO models are saved in a nested directory
        if backend == "openvino":
            save_dir = Path(save_dir) / backend
            # and we need to attach the file_suffix for both the .xml and .bin files
            shutil.move(save_dir / "openvino_model.xml", save_dir / file_name)
            shutil.move(save_dir / "openvino_model.bin", (save_dir / file_name).with_suffix(".bin"))
            save_dir = save_dir.as_posix()

        # Because we upload folders and save_dir now has unnecessary files (tokenizer.json, config.json, etc.),
        # we move the main file to a nested directory
        if backend == "onnx":
            dst_dir = Path(save_dir) / backend
            dst_dir.mkdir(parents=True, exist_ok=True)
            source = Path(save_dir) / file_name
            destination = dst_dir / file_name
            shutil.move(source, destination)
            save_dir = dst_dir.as_posix()

        if push_to_hub:
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
    model_kwargs={{"file_name": "{file_name}"}},
)

# Verify that everything works as expected
embeddings = model.encode(["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."])
print(embeddings.shape)

similarities = model.similarity(embeddings, embeddings)
print(similarities)
```
"""

            huggingface_hub.upload_folder(
                folder_path=save_dir,
                path_in_repo=backend,
                repo_id=model_name_or_path,
                repo_type="model",
                commit_message=f"Add exported {backend} model {file_name!r}",
                commit_description=commit_description,
                create_pr=create_pr,
            )

        else:
            dst_dir = Path(model_name_or_path) / backend
            # Create destination if it does not exist
            dst_dir.mkdir(parents=True, exist_ok=True)

            source = Path(save_dir) / file_name
            destination = dst_dir / file_name
            shutil.copy(source, destination)

            # OpenVINO has a second file to save: the .bin file
            if backend == "openvino":
                bin_source = (Path(save_dir) / file_name).with_suffix(".bin")
                bin_destination = (Path(dst_dir) / file_name).with_suffix(".bin")
                shutil.copy(bin_source, bin_destination)
