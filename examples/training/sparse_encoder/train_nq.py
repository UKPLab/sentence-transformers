from __future__ import annotations

import logging

from datasets import load_dataset

from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.sparse_encoder import SparseEncoder
from sentence_transformers.sparse_encoder.losses import CSRLoss
from sentence_transformers.sparse_encoder.models import CSRSparsity
from sentence_transformers.sparse_encoder.trainer import SparseEncoderTrainer
from sentence_transformers.sparse_encoder.training_args import (
    SparseEncoderTrainingArguments,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Initialize model components
    model_name = "sentence-transformers/all-mpnet-base-v2"
    transformer = Transformer(model_name)
    pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
    csr_sparsity = CSRSparsity(
        input_dim=transformer.get_word_embedding_dimension(),
        hidden_dim=transformer.get_word_embedding_dimension(),
        k=16,  # Number of top values to keep
        k_aux=512,  # Number of top values for auxiliary loss
    )

    # Create the SparseEncoder model
    model = SparseEncoder(modules=[transformer, pooling, csr_sparsity])

    # 2. Load the GooAQ dataset: https://huggingface.co/datasets/sentence-transformers/gooaq
    logging.info("Read the gooaq training dataset")
    full_dataset = load_dataset("sentence-transformers/gooaq", split="train").select(range(100_000))
    dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    logging.info(train_dataset)
    logging.info(eval_dataset)

    # Set up training arguments
    training_args = SparseEncoderTrainingArguments(
        output_dir="./sparse_encoder_nq",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        topk=16,
    )

    # Initialize the loss
    loss = CSRLoss(
        model=model,
        beta=1 / 32,  # Weight for auxiliary loss
        gamma=1,  # Weight for ranking loss
        scale=20.0,  # Scale for similarity computation
    )

    # Initialize trainer
    trainer = SparseEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )

    # Train model
    trainer.train()

    # Save model
    trainer.save_model("./sparse_encoder_nq_final")

    # Test the trained model
    test_sentences = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet in our solar system?",
    ]

    # Encode test sentences
    embeddings = model.encode(test_sentences, convert_to_sparse_tensor=True)

    # Get sparsity statistics
    stats = model.get_sparsity_stats(embeddings)
    logger.info(f"Sparsity statistics: {stats}")

    # Compute similarities
    similarities = model.similarity(embeddings, embeddings)
    logger.info(f"Similarity matrix:\n{similarities}")


if __name__ == "__main__":
    main()
