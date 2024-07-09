import pytest
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.losses.MatryoshkaLoss import MatryoshkaLoss
from unittest.mock import Mock


@pytest.fixture
def setup_model_and_data():
    model = Mock(spec=SentenceTransformer)

    def mock_forward(x):
        return {"sentence_embedding": torch.randn(2, 768)}

    model.side_effect = mock_forward

    matryoshka_dims = [768, 512, 256, 128, 64]
    matryoshka_weights = [1, 1, 1, 1, 1]
    loss = MultipleNegativesRankingLoss(model)

    anchors = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
    }
    positives = {
        "input_ids": torch.tensor([[7, 8, 9, 10], [11, 12, 13, 14]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]]),
    }
    features = [anchors, positives]
    # MultipleNegativesRankingLoss does not require labels
    labels = None

    return model, loss, matryoshka_dims, matryoshka_weights, features, labels


def test_loss(setup_model_and_data):
    model, loss, matryoshka_dims, matryoshka_weights, features, labels = setup_model_and_data

    matryoshka_loss = MatryoshkaLoss(model, loss, matryoshka_dims, matryoshka_weights)
    loss_value = matryoshka_loss(features, labels)

    assert isinstance(loss_value, torch.Tensor)
