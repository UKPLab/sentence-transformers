from __future__ import annotations

import logging
import os

from datasets import load_dataset

from sentence_transformers import SimilarityFunction
from sentence_transformers.sparse_encoder import (
    MLMTransformer,
    SparseEncoder,
    SpladePooling,
)
from sentence_transformers.sparse_encoder.evaluation.SparseNanoBEIREvaluator import SparseNanoBEIREvaluator
from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss
from sentence_transformers.sparse_encoder.trainer import SparseEncoderTrainer
from sentence_transformers.sparse_encoder.training_args import SparseEncoderTrainingArguments

# Set up logging
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def main():
    # Initialize the SPLADE model
    model_name = "answerdotai/ModernBERT-base"
    model = SparseEncoder(
        modules=[
            MLMTransformer(model_name),
            SpladePooling(pooling_strategy="max"),  # You can also use 'sum'
        ],
        device="cuda:0",
        similarity_fn_name=SimilarityFunction.DOT_PRODUCT,
    )
    model.eval()
    # 2a. Load the NQ dataset: https://huggingface.co/datasets/sentence-transformers/natural-questions
    logging.info("Read the Natural Questions training dataset")
    full_dataset = load_dataset("sentence-transformers/natural-questions", split="train").select(range(100_000))
    dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    logging.info(train_dataset)
    logging.info(eval_dataset)

    # 3. Initialize the loss
    loss = SpladeLoss(
        model=model,
        main_loss=SparseMultipleNegativesRankingLoss(model=model, scale=20, similarity_fct=model.similarity),
        lambda_query=0.1,  # Weight for query loss
        lambda_corpus=0.08,
    )  # Weight for document loss
    run_name = "splade-ModernBERT-nq-fresh"
    os.makedirs(f"runs/{run_name}", exist_ok=True)

    dev_evaluator = SparseNanoBEIREvaluator(["msmarco", "nfcorpus", "nq"], show_progress_bar=True, batch_size=16)
    os.makedirs(f"runs/{run_name}/eval", exist_ok=True)

    # Set up training arguments
    training_args = SparseEncoderTrainingArguments(
        output_dir=f"runs/{run_name}",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        bf16=True,
        logging_steps=200,
        eval_strategy="steps",
        eval_steps=2400,
        save_strategy="steps",
        save_steps=2400,
        learning_rate=5e-6,
        optim="adamw_torch",
        run_name=run_name,
        lr_scheduler_type="cosine",
        warmup_steps=0,
        lr_scheduler_kwargs={
            "num_cycles": 0.5,
        },
    )

    # Initialize trainer
    trainer = SparseEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )

    # Train model
    trainer.train()

    # 7. Evaluate the model performance again after training
    dev_evaluator(model, output_path=f"runs/{run_name}/eval", epoch=1)

    # 8. Save the trained & evaluated model locally
    os.makedirs(f"runs/{run_name}/final", exist_ok=True)
    model.save_pretrained(f"runs/{run_name}/final")

    model.push_to_hub(run_name)


if __name__ == "__main__":
    main()
