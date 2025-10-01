from __future__ import annotations

import warnings
from unittest.mock import Mock

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MatryoshkaLoss, MSELoss


@pytest.fixture
def mock_model():
    """Create a mock SentenceTransformer model for testing."""
    model = Mock(spec=SentenceTransformer)
    model.get_sentence_embedding_dimension.return_value = 768
    return model


def test_mse_loss_with_matching_dimensions(mock_model):
    """Test MSELoss with matching dimensions."""
    loss = MSELoss(model=mock_model)
    sentence_features = [{"input_ids": torch.tensor([[1, 2, 3]])}]
    x = torch.randn(1, 768)

    # Mock the model's forward method to return fixed embeddings
    mock_model.return_value = {"sentence_embedding": x}

    output = loss(sentence_features, x)
    assert isinstance(output, torch.Tensor)
    assert output.shape == torch.Size([])  # MSELoss returns a scalar
    mock_model.assert_called_once()


def test_mse_loss_with_dimension_mismatch(mock_model):
    """Test MSELoss with dimension mismatch."""
    loss = MSELoss(model=mock_model)
    sentence_features = [{"input_ids": torch.tensor([[1, 2, 3]])}]
    x = torch.randn(1, 768)  # Original dimension (teacher)

    # Mock the model's forward method to return fixed embeddings
    mock_model.return_value = {"sentence_embedding": torch.randn(1, 512)}

    output = loss(sentence_features, x)
    assert isinstance(output, torch.Tensor)
    assert output.shape == torch.Size([])  # MSELoss returns a scalar
    mock_model.assert_called_once()


def test_mse_loss_with_dimension_mismatch_target(mock_model):
    """Test MSELoss with teacher > student. This should crash"""
    loss = MSELoss(model=mock_model)
    sentence_features = [{"input_ids": torch.tensor([[1, 2, 3]])}]
    x = torch.randn(1, 512)  # Original dimension (teacher)

    # Mock the model's forward method to return fixed embeddings
    mock_model.return_value = {"sentence_embedding": torch.randn(1, 768)}

    with warnings.catch_warnings():
        # This function errors, but before it does so, it shows a warning.
        # We should not show it, as we handle the dimension mismatch internally.
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError):
            loss(sentence_features, x)


def test_mse_loss_matryoshka(mock_model):
    """Test MSELoss with multiple inputs (matryoshka)."""
    orig_loss = MSELoss(model=mock_model)
    loss = MatryoshkaLoss(model=mock_model, loss=orig_loss, matryoshka_dims=[32, 64, 128, 256, 512, 768])
    sentence_features = [{"input_ids": torch.tensor([[1, 2, 3]])}]
    x = torch.randn(1, 768)  # Original dimension (teacher)

    # Mock the model's forward method to return fixed embeddings
    mock_model.return_value = {"sentence_embedding": torch.randn(1, 768)}
    output = loss(sentence_features, x)
    assert isinstance(output, torch.Tensor)
    assert output.shape == torch.Size([])  # MSELoss returns a scalar
    assert mock_model.call_count == 6  # Called once for each matryoshka
