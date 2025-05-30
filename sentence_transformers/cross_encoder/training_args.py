from __future__ import annotations

from sentence_transformers.training_args import SentenceTransformerTrainingArguments


class CrossEncoderTrainingArguments(SentenceTransformerTrainingArguments):
    r"""
    CrossEncoderTrainingArguments extends :class:`~transformers.TrainingArguments` with additional arguments
    specific to Sentence Transformers. See :class:`~transformers.TrainingArguments` for the complete list of
    available arguments.

    Args:
        output_dir (`str`):
            The output directory where the model checkpoints will be written.
        prompts (`Union[Dict[str, Dict[str, str]], Dict[str, str], str]`, *optional*):
            The prompts to use for each column in the training, evaluation and test datasets. Four formats are accepted:

            1. `str`: A single prompt to use for all columns in the datasets, regardless of whether the training/evaluation/test
               datasets are :class:`datasets.Dataset` or a :class:`datasets.DatasetDict`.
            2. `Dict[str, str]`: A dictionary mapping column names to prompts, regardless of whether the training/evaluation/test
               datasets are :class:`datasets.Dataset` or a :class:`datasets.DatasetDict`.
            3. `Dict[str, str]`: A dictionary mapping dataset names to prompts. This should only be used if your training/evaluation/test
               datasets are a :class:`datasets.DatasetDict` or a dictionary of :class:`datasets.Dataset`.
            4. `Dict[str, Dict[str, str]]`: A dictionary mapping dataset names to dictionaries mapping column names to
               prompts. This should only be used if your training/evaluation/test datasets are a
               :class:`datasets.DatasetDict` or a dictionary of :class:`datasets.Dataset`.

        batch_sampler (Union[:class:`~sentence_transformers.training_args.BatchSamplers`, `str`], *optional*):
            The batch sampler to use. See :class:`~sentence_transformers.training_args.BatchSamplers` for valid options.
            Defaults to ``BatchSamplers.BATCH_SAMPLER``.
        multi_dataset_batch_sampler (Union[:class:`~sentence_transformers.training_args.MultiDatasetBatchSamplers`, `str`], *optional*):
            The multi-dataset batch sampler to use. See :class:`~sentence_transformers.training_args.MultiDatasetBatchSamplers`
            for valid options. Defaults to ``MultiDatasetBatchSamplers.PROPORTIONAL``.
        router_mapping (`Optional[Dict[str, str]]`, *optional*):
            A mapping of dataset names to Router routes, like "query" or "document". This is used to specify which
            Router module to use for each dataset. Two formats are accepted:

            1. `Dict[str, str]`: A mapping of dataset names to routes for single-dataset training/evaluation.
            2. `Dict[str, Dict[str, str]]`: A mapping of dataset names to a mapping of column names to routes for
               multi-dataset training/evaluation.
        learning_rate_mapping (`Optional[Dict[str, float]]`, *optional*):
            A mapping of parameter names to learning rates. This allows you to set different learning rates for
            different parts of the model, e.g., `{'IDF\.*': 1e-3}` for the IDF module. This is useful when you want to
            fine-tune specific parts of the model with different learning rates.
    """
