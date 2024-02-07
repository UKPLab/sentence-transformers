import pytest
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformer, losses
from datasets import DatasetDict


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
