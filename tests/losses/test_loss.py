from __future__ import annotations

import pytest
import torch
from datasets import Dataset
from torch import nn

from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import (
    CachedGISTEmbedLoss,
    CachedMultipleNegativesRankingLoss,
    GISTEmbedLoss,
    MultipleNegativesRankingLoss,
    TripletLoss,
)
from sentence_transformers.util import batch_to_device


@pytest.fixture(scope="module")
def guide_model():
    return SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")


def get_anchor_positive_negative_triplet():
    return {
        "losses": [
            (MultipleNegativesRankingLoss, {}),
            (CachedMultipleNegativesRankingLoss, {}),
            (TripletLoss, {}),
            (CachedGISTEmbedLoss, {"guide": "GUIDE_MODEL_PLACEHOLDER"}),
            (GISTEmbedLoss, {"guide": "GUIDE_MODEL_PLACEHOLDER"}),
        ],
        "correct": Dataset.from_dict(
            {
                "anchor": ["It's very sunny outside", "I love playing soccer", "I am a student"],
                "positive": ["The sun is out today", "I like playing soccer", "I am studying at university"],
                "negative": ["Data science is fun", "Cacti are beautiful", "Speakers are loud"],
            }
        ),
        "incorrect": Dataset.from_dict(
            {
                "anchor": ["It's very sunny outside", "I love playing soccer", "I am a student"],
                "positive": ["Data science is fun", "Cacti are beautiful", "Speakers are loud"],
                "negative": ["The sun is out today", "I like playing soccer", "I am studying at university"],
            }
        ),
    }


def get_loss_test_cases():
    anchor_positive_negative_triplet = get_anchor_positive_negative_triplet()
    return [
        (
            loss_class,
            loss_args,
            anchor_positive_negative_triplet["correct"],
            anchor_positive_negative_triplet["incorrect"],
        )
        for loss_class, loss_args in anchor_positive_negative_triplet["losses"]
    ]


def prepare_features_labels_from_dataset(model: SentenceTransformer, dataset: Dataset):
    device = model.device
    features = [
        batch_to_device(model.tokenize(dataset[column]), device)
        for column in dataset.column_names
        if column not in ["label", "score"]
    ]
    labels = None
    if "label" in dataset.column_names:
        labels = torch.tensor(dataset["label"]).to(device)
    elif "score" in dataset.column_names:
        labels = torch.tensor(dataset["score"]).to(device)
    return features, labels


def get_and_assert_loss_from_dataset(model: SentenceTransformer, loss_fn: nn.Module, dataset: Dataset):
    features, labels = prepare_features_labels_from_dataset(model, dataset)
    loss = loss_fn.forward(features, labels)
    assert isinstance(loss, torch.Tensor), f"Loss should be a torch.Tensor, but got {type(loss)}"
    assert loss.item() > 0, "Loss should be positive"
    assert loss.shape == (), "Loss should be a scalar"
    assert loss.requires_grad, "Loss should require gradients"
    return loss


@pytest.mark.parametrize("loss_class, loss_args, correct, incorrect", get_loss_test_cases())
def test_loss_function(
    stsb_bert_tiny_model_reused: SentenceTransformer, guide_model, loss_class, loss_args, correct, incorrect
):
    if "guide" in loss_args and loss_args["guide"] == "GUIDE_MODEL_PLACEHOLDER":
        loss_args["guide"] = guide_model

    model = stsb_bert_tiny_model_reused
    loss_fn = loss_class(model, **loss_args)
    correct_loss = get_and_assert_loss_from_dataset(model, loss_fn, correct)
    incorrect_loss = get_and_assert_loss_from_dataset(model, loss_fn, incorrect)

    assert correct_loss < incorrect_loss, "Loss should be lower for correct data than for incorrect data"
