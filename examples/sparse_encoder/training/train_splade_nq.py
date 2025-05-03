from __future__ import annotations

import logging
import os

from datasets import load_dataset

from sentence_transformers import SparseEncoder, SparseEncoderTrainer, SparseEncoderTrainingArguments
from sentence_transformers.sparse_encoder import evaluation, losses
from sentence_transformers.training_args import BatchSamplers

# Set up logging
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def main():
    # Initialize the SPLADE model
    model_name = "sparse-embedding/splade-distilbert-base-uncased-init"
    model = SparseEncoder(model_name)

    # 2a. Load the NQ dataset: https://huggingface.co/datasets/sentence-transformers/natural-questions
    logging.info("Read the Natural Questions training dataset")
    full_dataset = load_dataset("sentence-transformers/natural-questions", split="train").select(range(100_000))
    dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    logging.info(train_dataset)
    logging.info(eval_dataset)

    # 3. Initialize the loss
    lambda_query = 5e-5
    lambda_corpus = 3e-5

    loss = losses.SpladeLoss(
        model=model,
        main_loss=losses.SparseMultipleNegativesRankingLoss(model=model, scale=20, similarity_fct=model.similarity),
        lambda_query=lambda_query,  # Weight for query loss
        lambda_corpus=lambda_corpus,
    )  # Weight for document loss
    run_name = f"splade-distilbert-nq-fresh-lq{lambda_query}-lc{lambda_corpus}"
    os.makedirs(f"runs/{run_name}", exist_ok=True)

    dev_evaluator = evaluation.SparseNanoBEIREvaluator(show_progress_bar=True, batch_size=16)
    os.makedirs(f"runs/{run_name}/eval", exist_ok=True)

    # Set up training arguments
    training_args = SparseEncoderTrainingArguments(
        output_dir=f"runs/{run_name}",
        num_train_epochs=1,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=16,
        bf16=True,
        logging_steps=200,
        eval_strategy="steps",
        eval_steps=1650,
        save_strategy="steps",
        save_steps=1650,
        learning_rate=4e-5,
        run_name=run_name,
        seed=42,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoBEIR_mean_dot_ndcg@10",
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

    model.push_to_hub(f"sparse-embedding/{run_name}", private=True)


if __name__ == "__main__":
    main()
