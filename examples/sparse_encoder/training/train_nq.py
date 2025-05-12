from __future__ import annotations

import logging

from datasets import load_dataset

from sentence_transformers import SparseEncoder, SparseEncoderTrainer, SparseEncoderTrainingArguments
from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.sparse_encoder import evaluation, losses, models
from sentence_transformers.training_args import BatchSamplers

# Set up logging
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def main():
    # Initialize model components
    model_name = "microsoft/mpnet-base"
    transformer = Transformer(model_name)
    # transformer.requires_grad_(False)  # Freeze the transformer model
    pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
    csr_sparsity = models.CSRSparsity(
        input_dim=transformer.get_word_embedding_dimension(),
        hidden_dim=4 * transformer.get_word_embedding_dimension(),
        k=8,  # Number of top values to keep
        k_aux=512,  # Number of top values for auxiliary loss
    )

    # Create the SparseEncoder model
    model = SparseEncoder(modules=[transformer, pooling, csr_sparsity])

    # 2a. Load the NQ dataset: https://huggingface.co/datasets/sentence-transformers/natural-questions
    logging.info("Read the Natural Questions training dataset")
    full_dataset = load_dataset("sentence-transformers/natural-questions", split="train").select(range(100_000))
    dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    logging.info(train_dataset)
    logging.info(eval_dataset)

    # 3. Initialize the loss
    loss = losses.CSRLoss(
        model=model,
        beta=0.1,  # Weight for auxiliary loss
        gamma=1,  # Weight for ranking loss
        scale=20.0,  # Scale for similarity computation
    )

    # 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
    evaluators = []
    for k_dim in [16, 32, 64, 128, 256]:
        evaluators.append(evaluation.SparseNanoBEIREvaluator(["msmarco", "nfcorpus", "nq"], max_active_dims=k_dim))
    dev_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])
    dev_evaluator(model)

    # Set up training arguments
    run_name = "sparse-mpnet-base-nq-fresh"
    training_args = SparseEncoderTrainingArguments(
        output_dir=f"models/{run_name}",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        logging_steps=200,
        eval_strategy="steps",
        eval_steps=400,
        save_strategy="steps",
        save_steps=400,
        learning_rate=4e-5,
        optim="adamw_torch",
        weight_decay=1e-4,
        adam_epsilon=6.25e-10,
        run_name=run_name,
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
    dev_evaluator(model)

    # 8. Save the trained & evaluated model locally
    model.save_pretrained(f"models/{run_name}/final")

    model.push_to_hub(run_name)


if __name__ == "__main__":
    main()
