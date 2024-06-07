import json
import logging
import random
import re
from collections import Counter, defaultdict
from copy import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from platform import python_version
from textwrap import indent
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import transformers
from huggingface_hub import CardData, ModelCard
from huggingface_hub import dataset_info as get_dataset_info
from huggingface_hub import model_info as get_model_info
from huggingface_hub.repocard_data import EvalResult, eval_results_to_model_index
from huggingface_hub.utils import yaml_dump
from torch import nn
from tqdm.autonotebook import tqdm
from transformers import TrainerCallback
from transformers.integrations import CodeCarbonCallback
from transformers.modelcard import make_markdown_table
from transformers.trainer_callback import TrainerControl, TrainerState

from sentence_transformers import __version__ as sentence_transformers_version
from sentence_transformers.models import Transformer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.util import fullname, is_accelerate_available, is_datasets_available

if is_datasets_available():
    from datasets import Dataset, DatasetDict, Value

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
    from sentence_transformers.SentenceTransformer import SentenceTransformer
    from sentence_transformers.trainer import SentenceTransformerTrainer


class ModelCardCallback(TrainerCallback):
    def __init__(self, trainer: "SentenceTransformerTrainer", default_args_dict: Dict[str, Any]) -> None:
        super().__init__()
        self.trainer = trainer
        self.default_args_dict = default_args_dict

        callbacks = [
            callback
            for callback in self.trainer.callback_handler.callbacks
            if isinstance(callback, CodeCarbonCallback)
        ]
        if callbacks:
            trainer.model.model_card_data.code_carbon_callback = callbacks[0]

        trainer.model.model_card_data.trainer = trainer
        trainer.model.model_card_data.add_tags("generated_from_trainer")

    def on_init_end(
        self,
        args: SentenceTransformerTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: "SentenceTransformer",
        **kwargs,
    ) -> None:
        from sentence_transformers.losses import AdaptiveLayerLoss, Matryoshka2dLoss, MatryoshkaLoss

        # Try to infer the dataset "name", "id" and "revision" from the dataset cache files
        if self.trainer.train_dataset:
            model.model_card_data.train_datasets = model.model_card_data.extract_dataset_metadata(
                self.trainer.train_dataset, model.model_card_data.train_datasets, "train"
            )

        if self.trainer.eval_dataset:
            model.model_card_data.eval_datasets = model.model_card_data.extract_dataset_metadata(
                self.trainer.eval_dataset, model.model_card_data.eval_datasets, "eval"
            )

        if isinstance(self.trainer.loss, dict):
            losses = list(self.trainer.loss.values())
        else:
            losses = [self.trainer.loss]
        # Some losses are known to use other losses internally, e.g. MatryoshkaLoss, AdaptiveLayerLoss and Matryoshka2dLoss
        # So, verify for `loss` attributes in the losses
        loss_idx = 0
        while loss_idx < len(losses):
            loss = losses[loss_idx]
            if (
                isinstance(loss, (MatryoshkaLoss, AdaptiveLayerLoss, Matryoshka2dLoss))
                and hasattr(loss, "loss")
                and loss.loss not in losses
            ):
                losses.append(loss.loss)
            loss_idx += 1

        model.model_card_data.set_losses(losses)

    def on_train_begin(
        self,
        args: SentenceTransformerTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: "SentenceTransformer",
        **kwargs,
    ) -> None:
        # model.model_card_data.hyperparameters = extract_hyperparameters_from_trainer(self.trainer)
        ignore_keys = {
            "output_dir",
            "logging_dir",
            "logging_strategy",
            "logging_first_step",
            "logging_steps",
            "evaluation_strategy",
            "eval_steps",
            "eval_delay",
            "save_strategy",
            "save_steps",
            "save_total_limit",
            "metric_for_best_model",
            "greater_is_better",
            "report_to",
            "samples_per_label",
            "show_progress_bar",
            "do_train",
            "do_eval",
            "do_test",
            "run_name",
            "hub_token",
            "push_to_hub_token",
        }
        args_dict = args.to_dict()
        model.model_card_data.all_hyperparameters = {
            key: value for key, value in args_dict.items() if key not in ignore_keys
        }
        model.model_card_data.non_default_hyperparameters = {
            key: value
            for key, value in args_dict.items()
            if key not in ignore_keys and key in self.default_args_dict and value != self.default_args_dict[key]
        }

    def on_evaluate(
        self,
        args: SentenceTransformerTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: "SentenceTransformer",
        metrics: Dict[str, float],
        **kwargs,
    ) -> None:
        loss_dict = {" ".join(key.split("_")[1:]): metrics[key] for key in metrics if key.endswith("_loss")}
        if (
            model.model_card_data.training_logs
            and model.model_card_data.training_logs[-1]["Step"] == state.global_step
        ):
            model.model_card_data.training_logs[-1].update(loss_dict)
        else:
            model.model_card_data.training_logs.append(
                {
                    "Epoch": state.epoch,
                    "Step": state.global_step,
                    **loss_dict,
                }
            )

    def on_log(
        self,
        args: SentenceTransformerTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: "SentenceTransformer",
        logs: Dict[str, float],
        **kwargs,
    ) -> None:
        keys = {"loss"} & set(logs)
        if keys:
            if (
                model.model_card_data.training_logs
                and model.model_card_data.training_logs[-1]["Step"] == state.global_step
            ):
                model.model_card_data.training_logs[-1]["Training Loss"] = logs[keys.pop()]
            else:
                model.model_card_data.training_logs.append(
                    {
                        "Epoch": state.epoch,
                        "Step": state.global_step,
                        "Training Loss": logs[keys.pop()],
                    }
                )


YAML_FIELDS = [
    "language",
    "license",
    "library_name",
    "tags",
    "datasets",
    "metrics",
    "pipeline_tag",
    "widget",
    "model-index",
    "co2_eq_emissions",
    "base_model",
]
IGNORED_FIELDS = ["model", "trainer", "eval_results_dict"]


def get_versions() -> Dict[str, Any]:
    versions = {
        "python": python_version(),
        "sentence_transformers": sentence_transformers_version,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }
    if is_accelerate_available():
        from accelerate import __version__ as accelerate_version

        versions["accelerate"] = accelerate_version
    if is_datasets_available():
        from datasets import __version__ as datasets_version

        versions["datasets"] = datasets_version
    from tokenizers import __version__ as tokenizers_version

    versions["tokenizers"] = tokenizers_version

    return versions


@dataclass
class SentenceTransformerModelCardData(CardData):
    """A dataclass storing data used in the model card.

    Args:
        language (`Optional[Union[str, List[str]]]`): The model language, either a string or a list,
            e.g. "en" or ["en", "de", "nl"]
        license (`Optional[str]`): The license of the model, e.g. "apache-2.0", "mit",
            or "cc-by-nc-sa-4.0"
        model_name (`Optional[str]`): The pretty name of the model, e.g. "SentenceTransformer based on microsoft/mpnet-base".
        model_id (`Optional[str]`): The model ID when pushing the model to the Hub,
            e.g. "tomaarsen/sbert-mpnet-base-allnli".
        train_datasets (`List[Dict[str, str]]`): A list of the names and/or Hugging Face dataset IDs of the training datasets.
            e.g. [{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}, {"name": "STSB"}]
        eval_datasets (`List[Dict[str, str]]`): A list of the names and/or Hugging Face dataset IDs of the evaluation datasets.
            e.g. [{"name": "SNLI", "id": "stanfordnlp/snli"}, {"id": "mteb/stsbenchmark-sts"}]
        task_name (`str`): The human-readable task the model is trained on,
            e.g. "semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more".
        tags (`Optional[List[str]]`): A list of tags for the model,
            e.g. ["sentence-transformers", "sentence-similarity", "feature-extraction"].

    .. tip::

        Install `codecarbon <https://github.com/mlco2/codecarbon>`_ to automatically track carbon emission usage and
        include it in your model cards.

    Example::

        >>> model = SentenceTransformer(
        ...     "microsoft/mpnet-base",
        ...     model_card_data=SentenceTransformerModelCardData(
        ...         model_id="tomaarsen/sbert-mpnet-base-allnli",
        ...         train_datasets=[{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}],
        ...         eval_datasets=[{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}],
        ...         license="apache-2.0",
        ...         language="en",
        ...     ),
        ... )
    """

    # Potentially provided by the user
    language: Optional[Union[str, List[str]]] = field(default_factory=list)
    license: Optional[str] = None
    model_name: Optional[str] = None
    model_id: Optional[str] = None
    train_datasets: List[Dict[str, str]] = field(default_factory=list)
    eval_datasets: List[Dict[str, str]] = field(default_factory=list)
    task_name: str = (
        "semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more"
    )
    tags: Optional[List[str]] = field(
        default_factory=lambda: [
            "sentence-transformers",
            "sentence-similarity",
            "feature-extraction",
        ]
    )
    generate_widget_examples: Literal["deprecated"] = "deprecated"

    # Automatically filled by `ModelCardCallback` and the Trainer directly
    base_model: Optional[str] = field(default=None, init=False)
    base_model_revision: Optional[str] = field(default=None, init=False)
    non_default_hyperparameters: Dict[str, Any] = field(default_factory=dict, init=False)
    all_hyperparameters: Dict[str, Any] = field(default_factory=dict, init=False)
    eval_results_dict: Optional[Dict["SentenceEvaluator", Dict[str, Any]]] = field(default_factory=dict, init=False)
    training_logs: List[Dict[str, float]] = field(default_factory=list, init=False)
    widget: List[Dict[str, str]] = field(default_factory=list, init=False)
    predict_example: Optional[str] = field(default=None, init=False)
    label_example_list: List[Dict[str, str]] = field(default_factory=list, init=False)
    code_carbon_callback: Optional[CodeCarbonCallback] = field(default=None, init=False)
    citations: Dict[str, str] = field(default_factory=dict, init=False)
    best_model_step: Optional[int] = field(default=None, init=False)
    trainer: Optional["SentenceTransformerTrainer"] = field(default=None, init=False, repr=False)
    datasets: List[str] = field(default_factory=list, init=False, repr=False)

    # Utility fields
    first_save: bool = field(default=True, init=False)
    widget_step: int = field(default=-1, init=False)

    # Computed once, always unchanged
    pipeline_tag: str = field(default="sentence-similarity", init=False)
    library_name: str = field(default="sentence-transformers", init=False)
    version: Dict[str, str] = field(default_factory=get_versions, init=False)

    # Passed via `register_model` only
    model: Optional["SentenceTransformer"] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # We don't want to save "ignore_metadata_errors" in our Model Card
        infer_languages = not self.language
        if isinstance(self.language, str):
            self.language = [self.language]

        self.train_datasets = self.validate_datasets(self.train_datasets, infer_languages=infer_languages)
        self.eval_datasets = self.validate_datasets(self.eval_datasets, infer_languages=infer_languages)

        if self.model_id and self.model_id.count("/") != 1:
            logger.warning(
                f"The provided {self.model_id!r} model ID should include the organization or user,"
                ' such as "tomaarsen/mpnet-base-nli-matryoshka". Setting `model_id` to None.'
            )
            self.model_id = None

    def validate_datasets(self, dataset_list, infer_languages: bool = True) -> None:
        output_dataset_list = []
        for dataset in dataset_list:
            if "name" not in dataset:
                if "id" in dataset:
                    dataset["name"] = dataset["id"]

            if "id" in dataset:
                # Try to determine the language from the dataset on the Hub
                try:
                    info = get_dataset_info(dataset["id"])
                except Exception:
                    logger.warning(
                        f"The dataset `id` {dataset['id']!r} does not exist on the Hub. Setting the `id` to None."
                    )
                    del dataset["id"]
                else:
                    if info.cardData and infer_languages and "language" in info.cardData:
                        dataset_language = info.cardData.get("language")
                        if dataset_language is None:
                            break
                        if isinstance(dataset_language, str):
                            dataset_language = [dataset_language]
                        for language in dataset_language:
                            if language not in self.language:
                                self.language.append(language)

                    # Track dataset IDs for the metadata
                    if info.id not in self.datasets:
                        self.datasets.append(info.id)

            output_dataset_list.append(dataset)
        return output_dataset_list

    def set_losses(self, losses: List[nn.Module]) -> None:
        citations = {
            "Sentence Transformers": """
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
"""
        }
        for loss in losses:
            try:
                citations[loss.__class__.__name__] = loss.citation
            except Exception:
                pass
        inverted_citations = defaultdict(list)
        for loss, citation in citations.items():
            inverted_citations[citation].append(loss)

        def join_list(losses: List[str]) -> str:
            if len(losses) > 1:
                return ", ".join(losses[:-1]) + " and " + losses[-1]
            return losses[0]

        self.citations = {join_list(losses): citation for citation, losses in inverted_citations.items()}
        self.add_tags([f"loss:{loss}" for loss in {loss.__class__.__name__: loss for loss in losses}])

    def set_best_model_step(self, step: int) -> None:
        self.best_model_step = step

    def set_widget_examples(self, dataset: Union["Dataset", "DatasetDict"]) -> None:
        if isinstance(dataset, Dataset):
            dataset = DatasetDict(dataset=dataset)

        self.widget = []
        # Pick 5 random datasets to generate widget examples from
        dataset_names = Counter(random.choices(list(dataset.keys()), k=5))
        num_samples_to_check = 1000
        for dataset_name, num_samples in tqdm(
            dataset_names.items(), desc="Computing widget examples", unit="example", leave=False
        ):
            # Sample 1000 examples from the dataset, sort them by length, and pick the shortest examples as the core
            # examples for the widget
            columns = [
                column
                for column, feature in dataset[dataset_name].features.items()
                if isinstance(feature, Value) and feature.dtype == "string" and column != "dataset_name"
            ]
            str_dataset = dataset[dataset_name].select_columns(columns)
            dataset_size = len(str_dataset)
            lengths = {}
            for idx, sample in enumerate(
                str_dataset.select(random.sample(range(dataset_size), k=min(num_samples_to_check, dataset_size)))
            ):
                lengths[idx] = sum(len(value) for value in sample.values())

            indices, _ = zip(*sorted(lengths.items(), key=lambda x: x[1]))
            target_indices, backup_indices = indices[:num_samples], list(indices[num_samples:][::-1])

            # We want 4 texts, so we take texts from the backup indices, short texts first
            for idx in target_indices:
                # This is anywhere between 1 and n texts
                sentences = list(str_dataset[idx].values())
                while len(sentences) < 4 and backup_indices:
                    backup_idx = backup_indices.pop()
                    backup_sample = list(str_dataset[backup_idx].values())
                    if len(backup_sample) == 1:
                        # If there is only one text in the backup sample, we take it
                        sentences.extend(backup_sample)
                    else:
                        # Otherwise we prefer the 2nd text: the 1st can be another query
                        sentences.append(backup_sample[1])

                if len(sentences) < 4:
                    continue

                self.widget.append(
                    {"source_sentence": sentences[0], "sentences": random.sample(sentences[1:], k=len(sentences) - 1)}
                )
                self.predict_example = sentences[:3]

    def set_evaluation_metrics(self, evaluator: "SentenceEvaluator", metrics: Dict[str, Any]) -> None:
        from sentence_transformers.evaluation import SequentialEvaluator

        self.eval_results_dict[evaluator] = copy(metrics)

        # If the evaluator has a primary metric and we have a trainer, then add the primary metric to the training logs
        if hasattr(evaluator, "primary_metric") and (primary_metrics := evaluator.primary_metric):
            if isinstance(evaluator, SequentialEvaluator):
                primary_metrics = [sub_evaluator.primary_metric for sub_evaluator in evaluator.evaluators]
            elif isinstance(primary_metrics, str):
                primary_metrics = [primary_metrics]

            if self.trainer is None:
                step = 0
                epoch = 0
            else:
                step = self.trainer.state.global_step
                epoch = self.trainer.state.epoch
            training_log_metrics = {key: value for key, value in metrics.items() if key in primary_metrics}

            if self.training_logs and self.training_logs[-1]["Step"] == step:
                self.training_logs[-1].update(training_log_metrics)
            else:
                self.training_logs.append(
                    {
                        "Epoch": epoch,
                        "Step": step,
                        **training_log_metrics,
                    }
                )

    def set_label_examples(self, dataset: "Dataset") -> None:
        num_examples_per_label = 3
        examples = defaultdict(list)
        finished_labels = set()
        for sample in dataset:
            text = sample["text"]
            label = sample["label"]
            if label not in finished_labels:
                examples[label].append(f"<li>{repr(text)}</li>")
                if len(examples[label]) >= num_examples_per_label:
                    finished_labels.add(label)
            if len(finished_labels) == self.num_classes:
                break
        self.label_example_list = [
            {
                "Label": self.model.labels[label] if self.model.labels and isinstance(label, int) else label,
                "Examples": "<ul>" + "".join(example_set) + "</ul>",
            }
            for label, example_set in examples.items()
        ]

    def infer_datasets(
        self, dataset: Union["Dataset", "DatasetDict"], dataset_name: Optional[str] = None
    ) -> List[Dict[str, str]]:
        if isinstance(dataset, DatasetDict):
            return [
                dataset
                for dataset_name, sub_dataset in dataset.items()
                for dataset in self.infer_datasets(sub_dataset, dataset_name=dataset_name)
            ]

        def subtuple_finder(tuple: Tuple[str], subtuple: Tuple[str]) -> int:
            for i, element in enumerate(tuple):
                if element == subtuple[0] and tuple[i : i + len(subtuple)] == subtuple:
                    return i
            return -1

        cache_files = dataset.cache_files
        dataset_output = {}
        # Ignore the dataset name if it is a default name from the FitMixin backwards compatibility
        if dataset_name and re.match("_dataset_\d+", dataset_name):
            dataset_name = None
        if dataset_name:
            dataset_output["name"] = dataset_name
        if cache_files and "filename" in cache_files[0]:
            cache_path_parts = Path(cache_files[0]["filename"]).parts
            # Check if the cachefile is under "huggingface/datasets"
            subtuple = ("huggingface", "datasets")
            index = subtuple_finder(cache_path_parts, subtuple)
            if index == -1:
                return [dataset_output]

            # Get the folder after "huggingface/datasets"
            cache_dataset_name = cache_path_parts[index + len(subtuple)]
            # If the dataset has an author:
            if "___" in cache_dataset_name:
                author, dataset_name = cache_dataset_name.split("___")
                dataset_output["id"] = f"{author}/{dataset_name}"
            else:
                author = None
                dataset_name = cache_dataset_name
                # We can still be dealing with a local dataset here, so we wrap this with try-except
                try:
                    dataset_output["id"] = get_dataset_info(dataset_name).id
                except Exception:
                    # We can have a wide range of errors here, such as the dataset not existing, no internet, etc.
                    # So we use the generic Exception
                    pass

            # If the cache path ends with a 40 character hash, it is the current revision
            if len(cache_path_parts[-2]) == 40:
                dataset_output["revision"] = cache_path_parts[-2]

        return [dataset_output]

    def compute_dataset_metrics(
        self,
        dataset: Dict[str, str],
        dataset_info: Dict[str, Any],
        loss: Optional[Union[Dict[str, nn.Module], nn.Module]],
    ) -> Dict[str, str]:
        """
        Given a dataset, compute the following:
        * Dataset Size
        * Dataset Columns
        * Dataset Stats
            - Strings: min, mean, max word count/token length
            - Integers: Counter() instance
            - Floats: min, mean, max range
            - List: number of elements or min, mean, max number of elements
        * 3 Example samples
        * Loss function name
            - Loss function config
        """
        if not dataset:
            return {}

        dataset_info["size"] = len(dataset)
        dataset_info["columns"] = [f"<code>{column}</code>" for column in dataset.column_names]
        dataset_info["stats"] = {}
        for column in dataset.column_names:
            subsection = dataset[:1000][column]
            first = subsection[0]
            if isinstance(first, str):
                tokenized = self.model.tokenize(subsection)
                if isinstance(tokenized, dict) and "attention_mask" in tokenized:
                    lengths = tokenized["attention_mask"].sum(dim=1).tolist()
                    suffix = "tokens"
                else:
                    lengths = [len(sentence) for sentence in subsection]
                    suffix = "characters"
                dataset_info["stats"][column] = {
                    "dtype": "string",
                    "data": {
                        "min": f"{round(min(lengths), 2)} {suffix}",
                        "mean": f"{round(sum(lengths) / len(lengths), 2)} {suffix}",
                        "max": f"{round(max(lengths), 2)} {suffix}",
                    },
                }
            elif isinstance(first, (int, bool)):
                counter = Counter(subsection)
                dataset_info["stats"][column] = {
                    "dtype": "int",
                    "data": {
                        key: f"{'~' if len(counter) > 1 else ''}{counter[key] / len(subsection):.2%}"
                        for key in sorted(counter)
                    },
                }
            elif isinstance(first, float):
                dataset_info["stats"][column] = {
                    "dtype": "float",
                    "data": {
                        "min": round(min(dataset[column]), 2),
                        "mean": round(sum(dataset[column]) / len(dataset), 2),
                        "max": round(max(dataset[column]), 2),
                    },
                }
            elif isinstance(first, list):
                counter = Counter([len(lst) for lst in subsection])
                if len(counter) == 1:
                    dataset_info["stats"][column] = {
                        "dtype": "list",
                        "data": {
                            "size": f"{len(first)} elements",
                        },
                    }
                else:
                    dataset_info["stats"][column] = {
                        "dtype": "list",
                        "data": {
                            "min": f"{min(counter)} elements",
                            "mean": f"{sum(counter) / len(counter):.2f} elements",
                            "max": f"{max(counter)} elements",
                        },
                    }
            else:
                dataset_info["stats"][column] = {"dtype": fullname(first), "data": {}}

        def to_html_list(data: dict):
            return "<ul><li>" + "</li><li>".join(f"{key}: {value}" for key, value in data.items()) + "</li></ul>"

        stats_lines = [
            {"": "type", **{key: value["dtype"] for key, value in dataset_info["stats"].items()}},
            {"": "details", **{key: to_html_list(value["data"]) for key, value in dataset_info["stats"].items()}},
        ]
        dataset_info["stats_table"] = indent(make_markdown_table(stats_lines).replace("-:|", "--|"), "  ")

        dataset_info["examples"] = dataset[:3]
        num_samples = len(dataset_info["examples"][list(dataset_info["examples"])[0]])
        examples_lines = []
        for sample_idx in range(num_samples):
            columns = {}
            for column in dataset.column_names:
                value = dataset_info["examples"][column][sample_idx]
                # If the value is a long list, truncate it
                if isinstance(value, list) and len(value) > 5:
                    value = str(value[:5])[:-1] + ", ...]"
                # Avoid newlines in the table
                value = str(value).replace("\n", "<br>")
                columns[column] = f"<code>{value}</code>"
            examples_lines.append(columns)
        dataset_info["examples_table"] = indent(make_markdown_table(examples_lines).replace("-:|", "--|"), "  ")

        dataset_info["loss"] = {
            "fullname": fullname(loss),
        }
        if hasattr(loss, "get_config_dict"):
            config = loss.get_config_dict()
            try:
                str_config = json.dumps(config, indent=4)
            except TypeError:
                str_config = str(config)
            dataset_info["loss"]["config_code"] = indent(f"```json\n{str_config}\n```", "  ")
        return dataset_info

    def extract_dataset_metadata(
        self, dataset: Union["Dataset", "DatasetDict"], dataset_metadata, dataset_type: Literal["train", "eval"]
    ) -> Dict[str, Any]:
        if dataset:
            if dataset_metadata and (
                (isinstance(dataset, DatasetDict) and len(dataset_metadata) != len(dataset))
                or (isinstance(dataset, Dataset) and len(dataset_metadata) != 1)
            ):
                logger.warning(
                    f"The number of `{dataset_type}_datasets` in the model card data does not match the number of {dataset_type} datasets in the Trainer. "
                    f"Removing the provided `{dataset_type}_datasets` from the model card data."
                )
                dataset_metadata = []

            if not dataset_metadata:
                dataset_metadata = self.infer_datasets(dataset)

            if isinstance(dataset, DatasetDict):
                dataset_metadata = [
                    self.compute_dataset_metrics(
                        dataset_value,
                        dataset_info,
                        self.trainer.loss[dataset_name] if isinstance(self.trainer.loss, dict) else self.trainer.loss,
                    )
                    for dataset_name, dataset_value, dataset_info in zip(
                        dataset.keys(), dataset.values(), dataset_metadata
                    )
                ]
            else:
                dataset_metadata = [self.compute_dataset_metrics(dataset, dataset_metadata[0], self.trainer.loss)]

        # Try to get the number of training samples
        if dataset_type == "train":
            num_training_samples = sum([metadata.get("size", 0) for metadata in dataset_metadata])
            if num_training_samples:
                self.add_tags(f"dataset_size:{num_training_samples}")

        return self.validate_datasets(dataset_metadata)

    def register_model(self, model: "SentenceTransformer") -> None:
        self.model = model

    def set_model_id(self, model_id: str) -> None:
        self.model_id = model_id

    def set_base_model(self, model_id: str, revision: Optional[str] = None) -> None:
        try:
            model_info = get_model_info(model_id)
        except Exception:
            # Getting the model info can fail for many reasons: model does not exist, no internet, outage, etc.
            return False
        self.base_model = model_info.id
        if revision is None or revision == "main":
            revision = model_info.sha
        self.base_model_revision = revision
        return True

    def set_language(self, language: Union[str, List[str]]) -> None:
        if isinstance(language, str):
            language = [language]
        self.language = language

    def set_license(self, license: str) -> None:
        self.license = license

    def add_tags(self, tags: Union[str, List[str]]) -> None:
        if isinstance(tags, str):
            tags = [tags]
        for tag in tags:
            if tag not in self.tags:
                self.tags.append(tag)

    def try_to_set_base_model(self) -> None:
        if isinstance(self.model[0], Transformer):
            base_model = self.model[0].auto_model.config._name_or_path
            base_model_path = Path(base_model)
            # Sometimes the name_or_path ends exactly with the model_id, e.g.
            # "C:\\Users\\tom/.cache\\torch\\sentence_transformers\\BAAI_bge-small-en-v1.5\\"
            candidate_model_ids = ["/".join(base_model_path.parts[-2:])]
            # Sometimes the name_or_path its final part contains the full model_id, with "/" replaced with a "_", e.g.
            # "/root/.cache/torch/sentence_transformers/sentence-transformers_all-mpnet-base-v2/"
            # In that case, we take the last part, split on _, and try all combinations
            # e.g. "a_b_c_d" -> ['a/b_c_d', 'a_b/c_d', 'a_b_c/d']
            splits = base_model_path.name.split("_")
            candidate_model_ids += [
                "_".join(splits[:idx]) + "/" + "_".join(splits[idx:]) for idx in range(1, len(splits))
            ]
            for model_id in candidate_model_ids:
                if self.set_base_model(model_id):
                    break

    def format_eval_metrics(self) -> Dict[str, Any]:
        """Format the evaluation metrics for the model card.

        The following keys will be returned:
        - eval_metrics: A list of dictionaries containing the class name, description, dataset name, and a markdown table
          This is used to display the evaluation metrics in the model card.
        - metrics: A list of all metric keys. This is used in the model card metadata.
        - model-index: A list of dictionaries containing the task name, task type, dataset type, dataset name, metric name,
          metric type, and metric value. This is used to display the evaluation metrics in the model card metadata.
        """
        eval_metrics = []
        all_metrics = {}
        eval_results = []
        for evaluator, metrics in self.eval_results_dict.items():
            name = getattr(evaluator, "name", None)
            primary_metric = getattr(evaluator, "primary_metric", None)
            if name and all(key.startswith(name + "_") for key in metrics.keys()):
                metrics = {key[len(name) + 1 :]: value for key, value in metrics.items()}
                if primary_metric and primary_metric.startswith(name + "_"):
                    primary_metric = primary_metric[len(name) + 1 :]

            def try_to_pure_python(value: Any) -> Any:
                """Try to convert a value from a Numpy or Torch scalar to pure Python, if not already pure Python"""
                try:
                    if hasattr(value, "dtype"):
                        return value.item()
                except Exception:
                    pass
                return value

            metrics = {key: try_to_pure_python(value) for key, value in metrics.items()}

            table_lines = [
                {
                    "Metric": f"**{metric_key}**" if metric_key == primary_metric else metric_key,
                    "Value": f"**{round(metric_value, 4)}**"
                    if metric_key == primary_metric
                    else round(metric_value, 4),
                }
                for metric_key, metric_value in metrics.items()
            ]

            # E.g. "Binary Classification" or "Semantic Similarity"
            description = evaluator.description
            dataset_name = getattr(evaluator, "name", None)
            eval_metrics.append(
                {
                    "class_name": fullname(evaluator),
                    "description": description,
                    "dataset_name": dataset_name,
                    "table": make_markdown_table(table_lines).replace("-:|", "--|"),
                }
            )
            eval_results.extend(
                [
                    EvalResult(
                        task_name=description,
                        task_type=description.lower().replace(" ", "-"),
                        dataset_type=dataset_name or "unknown",
                        dataset_name=dataset_name.replace("_", " ").replace("-", " ") or "Unknown",
                        metric_name=metric_key.replace("_", " ").title(),
                        metric_type=metric_key,
                        metric_value=metric_value,
                    )
                    for metric_key, metric_value in metrics.items()
                    if isinstance(metric_value, (int, float))
                ]
            )
            all_metrics.update(metrics)

        return {
            "eval_metrics": eval_metrics,
            "metrics": list(all_metrics.keys()),
            "model-index": eval_results_to_model_index(self.model_name, eval_results),
        }

    def format_training_logs(self):
        # Get the keys from all evaluation lines
        eval_lines_keys = {key for lines in self.training_logs for key in lines.keys()}

        # Sort the metric columns: Epoch, Step, Training Loss, Validation Loss, Evaluator results
        def sort_metrics(key: str) -> str:
            if key == "Epoch":
                return "0"
            if key == "Step":
                return "1"
            if key == "Training Loss":
                return "2"
            if key.endswith("loss"):
                return "3"
            return key

        sorted_eval_lines_keys = sorted(eval_lines_keys, key=sort_metrics)
        training_logs = [
            {
                key: f"**{round(line[key], 4) if key in line else '-'}**"
                if line["Step"] == self.best_model_step
                else line.get(key, "-")
                for key in sorted_eval_lines_keys
            }
            for line in self.training_logs
        ]
        eval_lines = make_markdown_table(training_logs)
        return {
            "eval_lines": eval_lines,
            "explain_bold_in_eval": "**" in eval_lines,
        }

    def get_codecarbon_data(self) -> Dict[Literal["co2_eq_emissions"], Dict[str, Any]]:
        emissions_data = self.code_carbon_callback.tracker._prepare_emissions_data()
        results = {
            "co2_eq_emissions": {
                # * 1000 to convert kg to g
                "emissions": float(emissions_data.emissions) * 1000,
                "energy_consumed": float(emissions_data.energy_consumed),
                "source": "codecarbon",
                "training_type": "fine-tuning",
                "on_cloud": emissions_data.on_cloud == "Y",
                "cpu_model": emissions_data.cpu_model,
                "ram_total_size": emissions_data.ram_total_size,
                "hours_used": round(emissions_data.duration / 3600, 3),
            }
        }
        if emissions_data.gpu_model:
            results["co2_eq_emissions"]["hardware_used"] = emissions_data.gpu_model
        return results

    def to_dict(self) -> Dict[str, Any]:
        # Extract some meaningful examples from the evaluation or training dataset to showcase the performance
        if (
            not self.widget
            and self.trainer is not None
            and (dataset := self.trainer.eval_dataset or self.trainer.train_dataset)
        ):
            self.set_widget_examples(dataset)

        # Try to set the base model
        if self.first_save and not self.base_model:
            try:
                self.try_to_set_base_model()
            except Exception:
                pass

        # Set the model name
        if not self.model_name:
            if self.base_model:
                self.model_name = f"SentenceTransformer based on {self.base_model}"
            else:
                self.model_name = "SentenceTransformer"

        super_dict = {field.name: getattr(self, field.name) for field in fields(self)}

        # Compute required formats from the (usually post-training) evaluation data
        if self.eval_results_dict:
            try:
                super_dict.update(self.format_eval_metrics())
            except Exception as exc:
                logger.warning(f"Error while formatting evaluation metrics: {exc}")
                raise exc

        # Compute required formats for the during-training evaluation data
        if self.training_logs:
            try:
                super_dict.update(self.format_training_logs())
            except Exception as exc:
                logger.warning(f"Error while formatting training logs: {exc}")

        super_dict["hide_eval_lines"] = len(self.training_logs) > 100

        # Try to add the code carbon callback data
        if (
            self.code_carbon_callback
            and self.code_carbon_callback.tracker
            and self.code_carbon_callback.tracker._start_time is not None
        ):
            super_dict.update(self.get_codecarbon_data())

        # Add some additional metadata stored in the model itself
        super_dict["model_max_length"] = self.model.get_max_seq_length()
        super_dict["output_dimensionality"] = self.model.get_sentence_embedding_dimension()
        super_dict["model_string"] = str(self.model)
        if self.model.similarity_fn_name:
            super_dict["similarity_fn_name"] = {
                "cosine": "Cosine Similarity",
                "dot": "Dot Product",
                "euclidean": "Euclidean Distance",
                "manhattan": "Manhattan Distance",
            }.get(self.model.similarity_fn_name, self.model.similarity_fn_name.replace("_", " ").title())
        else:
            super_dict["similarity_fn_name"] = "Cosine Similarity"

        self.first_save = False

        for key in IGNORED_FIELDS:
            super_dict.pop(key, None)
        return super_dict

    def to_yaml(self, line_break=None) -> str:
        return yaml_dump(
            {key: value for key, value in self.to_dict().items() if key in YAML_FIELDS and value is not None},
            sort_keys=False,
            line_break=line_break,
        ).strip()


def generate_model_card(model: "SentenceTransformer") -> str:
    template_path = Path(__file__).parent / "model_card_template.md"
    model_card = ModelCard.from_template(card_data=model.model_card_data, template_path=template_path, hf_emoji="🤗")
    return model_card.content
