from __future__ import annotations

import logging

from datasets import load_dataset

from sentence_transformers import SparseEncoder, SparseEncoderTrainer, SparseEncoderTrainingArguments
from sentence_transformers.evaluation import SequentialEvaluator, SimilarityFunction
from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.sparse_encoder import evaluation, losses, models
from sentence_transformers.training_args import BatchSamplers

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Initialize model components
    model_name = "tomaarsen/mpnet-base-nli"
    transformer = Transformer(model_name)
    transformer.requires_grad_(False)  # Freeze the transformer model
    pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
    csr_sparsity = models.CSRSparsity(
        input_dim=transformer.get_word_embedding_dimension(),
        hidden_dim=4 * transformer.get_word_embedding_dimension(),
        k=256,  # Number of top values to keep
        k_aux=512,  # Number of top values for auxiliary loss
    )

    # Create the SparseEncoder model
    model = SparseEncoder(modules=[transformer, pooling, csr_sparsity])

    output_dir = "examples/sparse_encoder/output/sparse_encoder_nli_frozen_transformer_from_pretrained"
    # 2. Load the AllNLI dataset: https://huggingface.co/datasets/sentence-transformers/all-nli
    train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
    eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
    logging.info(train_dataset)

    # 3. Initialize the loss
    loss = losses.CSRLoss(
        model=model,
        beta=0.1,  # Weight for auxiliary loss
        gamma=1,  # Weight for ranking loss
        scale=20.0,  # Scale for similarity computation
    )

    # 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
    stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
    evaluators = []
    for k_dim in [16, 32, 64, 128, 256]:
        evaluators.append(
            evaluation.SparseEmbeddingSimilarityEvaluator(
                sentences1=stsb_eval_dataset["sentence1"],
                sentences2=stsb_eval_dataset["sentence2"],
                scores=stsb_eval_dataset["score"],
                main_similarity=SimilarityFunction.COSINE,
                name=f"sts-dev-{k_dim}",
                max_active_dims=k_dim,
            )
        )
    dev_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])

    # Set up training arguments
    training_args = SparseEncoderTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        learning_rate=4e-5,
        optim="adamw_torch",
        weight_decay=1e-4,
        adam_epsilon=6.25e-10,
        run_name="sparse_encoder_nli_frozen_transformer_from_pretrained",
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

    # 7. Evaluate the model performance on the STS Benchmark test dataset
    test_dataset = load_dataset("sentence-transformers/stsb", split="test")
    evaluators = []
    for k_dim in [16, 32, 64, 128, 256]:
        evaluators.append(
            evaluation.SparseEmbeddingSimilarityEvaluator(
                sentences1=test_dataset["sentence1"],
                sentences2=test_dataset["sentence2"],
                scores=test_dataset["score"],
                main_similarity=SimilarityFunction.COSINE,
                name=f"sts-test-{k_dim}",
                truncate_dim=k_dim,
            )
        )
    test_evaluator = SequentialEvaluator(evaluators)
    test_evaluator(model)

    # 8. Save the trained & evaluated model locally
    model.save(output_dir)


if __name__ == "__main__":
    main()
