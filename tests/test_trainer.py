from __future__ import annotations

import re
import tempfile
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.util import is_datasets_available, is_training_available
from tests.utils import SafeTemporaryDirectory

if is_datasets_available():
    from datasets import DatasetDict, load_dataset


@pytest.mark.skipif(
    not is_training_available(),
    reason='Sentence Transformers was not installed with the `["train"]` extra.',
)
def test_trainer_multi_dataset_errors(
    stsb_bert_tiny_model: SentenceTransformer, stsb_dataset_dict: DatasetDict
) -> None:
    train_dataset = stsb_dataset_dict["train"]
    loss = {
        "multi_nli": losses.CosineSimilarityLoss(model=stsb_bert_tiny_model),
        "snli": losses.CosineSimilarityLoss(model=stsb_bert_tiny_model),
        "stsb": losses.CosineSimilarityLoss(model=stsb_bert_tiny_model),
    }
    with pytest.raises(
        ValueError, match="If the provided `loss` is a dict, then the `train_dataset` must be a `DatasetDict`."
    ):
        SentenceTransformerTrainer(model=stsb_bert_tiny_model, train_dataset=train_dataset, loss=loss)

    train_dataset = DatasetDict(
        {
            "multi_nli": stsb_dataset_dict["train"],
            "snli": stsb_dataset_dict["train"],
            "stsb": stsb_dataset_dict["train"],
            "stsb-extra": stsb_dataset_dict["train"],
        }
    )
    with pytest.raises(
        ValueError,
        match="If the provided `loss` is a dict, then all keys from the `train_dataset` dictionary must occur in `loss` also. "
        r"Currently, \['stsb-extra'\] occurs in `train_dataset` but not in `loss`.",
    ):
        SentenceTransformerTrainer(model=stsb_bert_tiny_model, train_dataset=train_dataset, loss=loss)

    train_dataset = DatasetDict(
        {
            "multi_nli": stsb_dataset_dict["train"],
            "snli": stsb_dataset_dict["train"],
            "stsb": stsb_dataset_dict["train"],
        }
    )
    with pytest.raises(
        ValueError, match="If the provided `loss` is a dict, then the `eval_dataset` must be a `DatasetDict`."
    ):
        SentenceTransformerTrainer(
            model=stsb_bert_tiny_model,
            train_dataset=train_dataset,
            eval_dataset=stsb_dataset_dict["validation"],
            loss=loss,
        )

    eval_dataset = DatasetDict(
        {
            "multi_nli": stsb_dataset_dict["validation"],
            "snli": stsb_dataset_dict["validation"],
            "stsb": stsb_dataset_dict["validation"],
            "stsb-extra-1": stsb_dataset_dict["validation"],
            "stsb-extra-2": stsb_dataset_dict["validation"],
        }
    )
    with pytest.raises(
        ValueError,
        match="If the provided `loss` is a dict, then all keys from the `eval_dataset` dictionary must occur in `loss` also. "
        r"Currently, \['stsb-extra-1', 'stsb-extra-2'\] occur in `eval_dataset` but not in `loss`.",
    ):
        SentenceTransformerTrainer(
            model=stsb_bert_tiny_model, train_dataset=train_dataset, eval_dataset=eval_dataset, loss=loss
        )


@pytest.mark.skipif(
    not is_training_available(),
    reason='Sentence Transformers was not installed with the `["train"]` extra.',
)
def test_trainer_invalid_column_names(
    stsb_bert_tiny_model: SentenceTransformer, stsb_dataset_dict: DatasetDict
) -> None:
    train_dataset = stsb_dataset_dict["train"]
    for column_name in ("return_loss", "dataset_name"):
        invalid_train_dataset = train_dataset.rename_column("sentence1", column_name)
        trainer = SentenceTransformerTrainer(model=stsb_bert_tiny_model, train_dataset=invalid_train_dataset)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"The following column names are invalid in your dataset: ['{column_name}']."
                " Avoid using these column names, as they are reserved for internal use."
            ),
        ):
            trainer.train()

        invalid_train_dataset = DatasetDict(
            {
                "stsb": train_dataset.rename_column("sentence1", column_name),
                "stsb-2": train_dataset,
            }
        )
        trainer = SentenceTransformerTrainer(model=stsb_bert_tiny_model, train_dataset=invalid_train_dataset)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"The following column names are invalid in your stsb dataset: ['{column_name}']."
                " Avoid using these column names, as they are reserved for internal use."
            ),
        ):
            trainer.train()


@pytest.mark.skipif(
    not is_training_available(),
    reason='Sentence Transformers was not installed with the `["train"]` extra.',
)
def test_model_card_reuse(stsb_bert_tiny_model: SentenceTransformer):
    assert stsb_bert_tiny_model._model_card_text
    # Reuse the model card if no training was done
    with SafeTemporaryDirectory() as tmp_folder:
        model_path = Path(tmp_folder) / "tiny_model_local"
        stsb_bert_tiny_model.save(str(model_path))

        with open(model_path / "README.md") as f:
            model_card_text = f.read()
        assert model_card_text == stsb_bert_tiny_model._model_card_text

    # Create a new model card if a Trainer was initialized
    SentenceTransformerTrainer(model=stsb_bert_tiny_model)

    with SafeTemporaryDirectory() as tmp_folder:
        model_path = Path(tmp_folder) / "tiny_model_local"
        stsb_bert_tiny_model.save(str(model_path))

        with open(model_path / "README.md") as f:
            model_card_text = f.read()
        assert model_card_text != stsb_bert_tiny_model._model_card_text


@pytest.mark.skipif(
    not is_training_available(),
    reason='Sentence Transformers was not installed with the `["train"]` extra.',
)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("train_dict", [False, True])
@pytest.mark.parametrize("eval_dict", [False, True])
@pytest.mark.parametrize("loss_dict", [False, True])
def test_trainer(
    stsb_bert_tiny_model: SentenceTransformer, streaming: bool, train_dict: bool, eval_dict: bool, loss_dict: bool
) -> None:
    """
    Some cases are not allowed:
    * streaming=True and train_dict=True: streaming is not supported with DatasetDict, because our DatasetDict
      implementation concatenates the individual datasets and uses their sizes for tracking which original dataset the samples are from.
      This is not possible with streaming datasets as they don't have a known size.
      (Note: streaming=True and eval_dict=True does not throw an error because the transformers Trainer already allows for
      dictionaries of evaluation datasets. In that case, the evaluation dataloader is created with just a normal IterableDataset multiple
      times instead of a ConcatDataset of IterableDatasets.)
    * loss_dict=True and (train_dict=False or eval_dict=False): if loss is a dict, then train_dataset and eval_dataset must be dicts too,
      otherwise the trainer doesn't know which loss to use.
    """
    context = nullcontext()
    if loss_dict and not train_dict:
        context = pytest.raises(
            ValueError, match="If the provided `loss` is a dict, then the `train_dataset` must be a `DatasetDict`."
        )
    elif loss_dict and not eval_dict:
        context = pytest.raises(
            ValueError, match="If the provided `loss` is a dict, then the `eval_dataset` must be a `DatasetDict`."
        )
    elif streaming and train_dict:
        context = pytest.raises(
            ValueError,
            match="Sentence Transformers is not compatible with a DatasetDict containing an IterableDataset.",
        )

    model = stsb_bert_tiny_model
    original_model = deepcopy(model)
    train_dataset = load_dataset("sentence-transformers/stsb", split="train[:10]")
    eval_dataset = load_dataset("sentence-transformers/stsb", split="validation[:10]")
    loss = losses.CosineSimilarityLoss(model=model)

    if streaming:
        train_dataset = train_dataset.to_iterable_dataset()
        eval_dataset = eval_dataset.to_iterable_dataset()
    if train_dict:
        train_dataset = DatasetDict({"stsb-1": train_dataset, "stsb-2": train_dataset})
    if eval_dict:
        eval_dataset = DatasetDict({"stsb-1": eval_dataset, "stsb-2": eval_dataset})
    if loss_dict:
        loss = {
            "stsb-1": loss,
            "stsb-2": loss,
        }

    with tempfile.TemporaryDirectory() as temp_dir:
        args = SentenceTransformerTrainingArguments(
            output_dir=str(temp_dir),
            max_steps=2,
            eval_steps=2,
            eval_strategy="steps",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
        )
        with context:
            trainer = SentenceTransformerTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                loss=loss,
            )
            trainer.train()

    if isinstance(context, nullcontext):
        original_embeddings = original_model.encode("The cat is on the mat.", convert_to_tensor=True)
        new_embeddings = model.encode("The cat is on the the mat.", convert_to_tensor=True)
        assert not torch.equal(original_embeddings, new_embeddings)
