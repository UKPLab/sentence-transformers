import re
import tempfile
from pathlib import Path

import pytest

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.util import is_datasets_available, is_training_available

if is_datasets_available():
    from datasets import DatasetDict


@pytest.mark.skipif(
    not is_training_available(),
    reason='Sentence Transformers was not installed with the `["train"]` extra.',
)
def test_trainer_multi_dataset_errors(
    stsb_bert_tiny_model: SentenceTransformer, stsb_dataset_dict: "DatasetDict"
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
        "Currently, \['stsb-extra'\] occurs in `train_dataset` but not in `loss`.",
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
        "Currently, \['stsb-extra-1', 'stsb-extra-2'\] occur in `eval_dataset` but not in `loss`.",
    ):
        SentenceTransformerTrainer(
            model=stsb_bert_tiny_model, train_dataset=train_dataset, eval_dataset=eval_dataset, loss=loss
        )


@pytest.mark.skipif(
    not is_training_available(),
    reason='Sentence Transformers was not installed with the `["train"]` extra.',
)
def test_trainer_invalid_column_names(
    stsb_bert_tiny_model: SentenceTransformer, stsb_dataset_dict: "DatasetDict"
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
    with tempfile.TemporaryDirectory() as tmp_folder:
        model_path = Path(tmp_folder) / "tiny_model_local"
        stsb_bert_tiny_model.save(str(model_path))

        with open(model_path / "README.md", "r") as f:
            model_card_text = f.read()
        assert model_card_text == stsb_bert_tiny_model._model_card_text

    # Create a new model card if a Trainer was initialized
    SentenceTransformerTrainer(model=stsb_bert_tiny_model)

    with tempfile.TemporaryDirectory() as tmp_folder:
        model_path = Path(tmp_folder) / "tiny_model_local"
        stsb_bert_tiny_model.save(str(model_path))

        with open(model_path / "README.md", "r") as f:
            model_card_text = f.read()
        assert model_card_text != stsb_bert_tiny_model._model_card_text
