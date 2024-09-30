from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import huggingface_hub

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

    try:
        from optimum.onnxruntime.configuration import OptimizationConfig
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
        raise ValueError('The model must be a SentenceTransformer model loaded with `backend="onnx"`.')

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

    if push_to_hub:
        with tempfile.TemporaryDirectory() as save_dir:
            optimizer.optimize(
                save_dir=save_dir,
                file_suffix=file_suffix,
                optimization_config=optimization_config,
            )
            file_name = f"model_{file_suffix}.onnx"
            source = (Path(save_dir) / file_name).as_posix()
            destination = Path(file_name).as_posix()

            commit_description = ""
            if create_pr:
                opt_config_string = (
                    str(optimization_config).replace("(", "(\n\t").replace(", ", ",\n\t").replace(")", "\n)")
                )
                commit_description = f"""\
Hello!

*This pull request has been automatically generated from the `export_optimized_onnx_model` function from the Sentence Transformers library.*

## Optimization Config
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
    backend="onnx",
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
                path_in_repo=destination,
                repo_id=model_name_or_path,
                repo_type="model",
                commit_message=f"Add optimized ONNX model {file_name!r}",
                commit_description=commit_description,
                create_pr=create_pr,
            )

    else:
        with tempfile.TemporaryDirectory() as save_dir:
            optimizer.optimize(
                save_dir=save_dir,
                file_suffix=file_suffix,
                optimization_config=optimization_config,
            )

            file_name = f"model_{file_suffix}.onnx"
            source = os.path.join(save_dir, file_name)
            destination = os.path.join(model_name_or_path, file_name)
            # Create destination if it does not exist
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy(source, destination)
