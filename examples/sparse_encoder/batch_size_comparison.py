import gc
import time

import torch
from datasets import Dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss import (
    SparseMultipleNegativesRankingLoss,
)
from sentence_transformers.sparse_encoder.models import MLMTransformer, SpladePooling
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder
from sentence_transformers.sparse_encoder.trainer import SparseEncoderTrainer
from sentence_transformers.sparse_encoder.training_args import (
    SparseEncoderTrainingArguments,
)


def create_dataset(num_samples=1024):
    """Create a synthetic dataset with the specified number of samples."""
    anchors = [f"This is sample {i} for testing batch size capabilities." for i in range(num_samples)]
    positives = [f"This is the positive example for sample {i}." for i in range(num_samples)]

    return Dataset.from_dict(
        {
            "anchor": anchors,
            "positive": positives,
        }
    )


def test_batch_size(model, loss_fn, dataset, start_batch_size=8, max_batch_size=1024, step=2):
    """
    Test what batch size the model can handle without OOM errors.

    Args:
        model: The model to test
        loss_fn: The loss function to use
        dataset: The dataset to use
        start_batch_size: The starting batch size to test
        max_batch_size: The maximum batch size to test
        step: The step size to increase the batch size by

    Returns:
        The maximum batch size that works without OOM
    """
    batch_size = start_batch_size
    max_working_batch_size = start_batch_size

    while batch_size <= max_batch_size:
        print(f"Testing batch size: {batch_size}")
        try:
            # Clear memory
            gc.collect()
            torch.cuda.empty_cache()

            # Create appropriate trainer based on model type
            if isinstance(model, SparseEncoder):
                training_args = SparseEncoderTrainingArguments(
                    num_train_epochs=1,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                )
                trainer = SparseEncoderTrainer(
                    model=model,
                    train_dataset=dataset.select(range(0, batch_size)),
                    loss=loss_fn,
                    args=training_args,
                )
            else:
                training_args = SentenceTransformerTrainingArguments(
                    num_train_epochs=1,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                )
                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=dataset.select(range(0, batch_size)),
                    loss=loss_fn,
                    args=training_args,
                )

            # Try to train for a few steps
            start_time = time.time()
            trainer.train()  # Just run 5 steps to test
            elapsed_time = time.time() - start_time

            print(f"✅ Batch size {batch_size} works! Time per step: {elapsed_time / 5:.2f}s")
            max_working_batch_size = batch_size

            # Increase batch size for next test
            batch_size *= step

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ OOM error with batch size {batch_size}")
                break
            else:
                print(f"❌ Error with batch size {batch_size}: {e}")
                break
        except Exception as e:
            print(f"❌ Unexpected error with batch size {batch_size}: {e}")
            break

    return max_working_batch_size


def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create a dataset with 10,000 samples
    dataset = create_dataset(10000)
    print(f"Created dataset with {len(dataset)} samples")

    # Test 1: Standard BERT model with MRL loss
    print("\n" + "=" * 50)
    print("TEST 1: Standard BERT model with MRL loss")
    print("=" * 50)

    # Initialize standard model
    standard_model = SentenceTransformer("tomaarsen/bert-base-nq-prompts", device=device)
    standard_model.train()
    # print number of parameters trainable
    num_params = sum(p.numel() for p in standard_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in standard model: {num_params}")
    standard_loss = losses.MultipleNegativesRankingLoss(standard_model)

    # Test batch size
    max_standard_batch_size = test_batch_size(standard_model, standard_loss, dataset)
    print(f"\nMaximum working batch size for standard model: {max_standard_batch_size}")

    # Test 2: Sparse BERT model with Sparse MRL loss
    print("\n" + "=" * 50)
    print("TEST 2: Sparse BERT model with Sparse MRL loss")
    print("=" * 50)

    # Initialize sparse model
    model_name = "naver/splade-cocondenser-ensembledistil"
    sparse_model = SparseEncoder(
        modules=[
            MLMTransformer(model_name),
            SpladePooling(pooling_strategy="max"),
        ],
        device=device,
    )
    sparse_model.train()
    # print number of parameters trainable
    num_params = sum(p.numel() for p in sparse_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in sparse model: {num_params}")
    sparse_loss = SparseMultipleNegativesRankingLoss(sparse_model)

    # Test batch size
    max_sparse_batch_size = test_batch_size(sparse_model, sparse_loss, dataset)
    print(f"\nMaximum working batch size for sparse model: {max_sparse_batch_size}")

    # Print comparison
    print("\n" + "=" * 50)
    print("BATCH SIZE COMPARISON")
    print("=" * 50)
    print(f"Standard BERT model: {max_standard_batch_size}")
    print(f"Sparse BERT model: {max_sparse_batch_size}")
    print(f"Memory efficiency improvement: {max_sparse_batch_size / max_standard_batch_size:.2f}x")


if __name__ == "__main__":
    main()
