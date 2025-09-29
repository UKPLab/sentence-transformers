from __future__ import annotations

import warnings
from unittest.mock import Mock

import pytest

from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss


@pytest.fixture
def mock_model():
    """Create a mock SentenceTransformer model for testing."""
    model = Mock(spec=SentenceTransformer)
    model.get_sentence_embedding_dimension.return_value = 768
    return model


@pytest.fixture
def mock_loss():
    """Create a mock base loss for testing."""
    return Mock(spec=MultipleNegativesRankingLoss)


def test_empty_matryoshka_dims_raises_error(mock_model, mock_loss):
    """Test that empty matryoshka_dims raises ValueError."""
    with pytest.raises(ValueError, match="You must provide at least one dimension in matryoshka_dims."):
        MatryoshkaLoss(model=mock_model, loss=mock_loss, matryoshka_dims=[])


def test_zero_dimension_raises_error(mock_model, mock_loss):
    """Test that zero dimension in matryoshka_dims raises ValueError."""
    with pytest.raises(ValueError, match="All dimensions passed to a matryoshka loss must be > 0."):
        MatryoshkaLoss(model=mock_model, loss=mock_loss, matryoshka_dims=[512, 0, 256])


def test_negative_dimension_raises_error(mock_model, mock_loss):
    """Test that negative dimension in matryoshka_dims raises ValueError."""
    with pytest.raises(ValueError, match="All dimensions passed to a matryoshka loss must be > 0."):
        MatryoshkaLoss(model=mock_model, loss=mock_loss, matryoshka_dims=[512, -100, 256])


def test_weights_length_mismatch_raises_error(mock_model, mock_loss):
    """Test that length mismatch between weights and dims raises ValueError."""
    with pytest.raises(ValueError, match="matryoshka_weights must be the same length as matryoshka_dims."):
        MatryoshkaLoss(
            model=mock_model,
            loss=mock_loss,
            matryoshka_dims=[512, 256, 128],
            matryoshka_weights=[1.0, 0.5],  # Only 2 weights for 3 dims
        )


def test_dimension_exceeds_model_dimension_raises_error(mock_model, mock_loss):
    """Test that dimension exceeding model's embedding dimension raises ValueError."""
    expected_msg = "Dimensions in matryoshka_dims cannot exceed the model's embedding dimension of 768."
    with pytest.raises(ValueError, match=expected_msg):
        MatryoshkaLoss(
            model=mock_model,
            loss=mock_loss,
            matryoshka_dims=[512, 1024, 256],  # 1024 > 768
        )


def test_model_dimension_not_in_dims_warns(mock_model, mock_loss, caplog):
    """Test that model dimension not in matryoshka_dims produces warning."""
    with caplog.at_level("WARNING"):
        MatryoshkaLoss(
            model=mock_model,
            loss=mock_loss,
            matryoshka_dims=[512, 256, 128],  # 768 not included
        )

    # Check that warning was logged
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    expected_msg = (
        "The model's embedding dimension 768 is not included in matryoshka_dims: [512, 256, 128]. "
        "This means that the full model dimension won't be trained, which may lead to degraded performance "
        "when using the model without specifying a lower truncation dimension. It is strongly recommended to include "
        "768 in matryoshka_dims."
    )
    assert caplog.records[0].message == expected_msg


def test_model_dimension_none_no_validation(mock_loss):
    """Test that when model dimension is None, no validation or warning occurs."""
    model = Mock(spec=SentenceTransformer)
    model.get_sentence_embedding_dimension.return_value = None

    # This should not raise any error or warning, even with large dimensions
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        MatryoshkaLoss(
            model=model,
            loss=mock_loss,
            matryoshka_dims=[2048, 1024, 512],  # Large dimensions, but no validation
        )


def test_valid_initialization_no_warnings(mock_model, mock_loss):
    """Test that valid initialization with model dimension included produces no warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        MatryoshkaLoss(
            model=mock_model,
            loss=mock_loss,
            matryoshka_dims=[768, 512, 256, 128],  # 768 included
        )


def test_valid_initialization_with_weights(mock_model, mock_loss):
    """Test that valid initialization with custom weights works correctly."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        loss = MatryoshkaLoss(
            model=mock_model, loss=mock_loss, matryoshka_dims=[768, 512, 256], matryoshka_weights=[1.0, 0.8, 0.6]
        )

        # Verify that dimensions and weights are sorted in descending order
        assert loss.matryoshka_dims == (768, 512, 256)
        assert loss.matryoshka_weights == (1.0, 0.8, 0.6)


def test_dimensions_sorted_descending(mock_model, mock_loss):
    """Test that dimensions and weights are sorted in descending order."""
    loss = MatryoshkaLoss(
        model=mock_model,
        loss=mock_loss,
        matryoshka_dims=[256, 768, 512],  # Unsorted
        matryoshka_weights=[0.6, 1.0, 0.8],  # Corresponding weights
    )

    # Should be sorted in descending order
    assert loss.matryoshka_dims == (768, 512, 256)
    assert loss.matryoshka_weights == (1.0, 0.8, 0.6)


def test_default_weights_when_none(mock_model, mock_loss):
    """Test that default weights (all 1s) are used when weights is None."""
    loss = MatryoshkaLoss(model=mock_model, loss=mock_loss, matryoshka_dims=[768, 512, 256], matryoshka_weights=None)

    assert loss.matryoshka_weights == (1, 1, 1)
