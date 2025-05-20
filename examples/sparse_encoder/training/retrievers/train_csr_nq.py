"""
This example trains a SparseEncoder for the Natural Questions (NQ) dataset.
The training script fine-tunes a SparseEncoder using the CSR loss function for retrieval.
It loads a subset of the Natural Questions dataset, splits it into training and evaluation subsets,
and trains the model as a retriever. After training, the model is evaluated and saved locally,
with an optional step to push the trained model to the Hugging Face Hub.

Usage:
python train_csr_nq.py
"""

import logging
import traceback

from datasets import load_dataset

from sentence_transformers import (
    SparseEncoder,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.sparse_encoder import evaluation, losses
from sentence_transformers.training_args import BatchSamplers

# Set up logging
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def main():
    model_name = "mixedbread-ai/mxbai-embed-large-v1"

    train_batch_size = 64
    num_epochs = 1

    # 1a. Load a model to finetune with 1b. (Optional) model card data
    model = SparseEncoder(
        model_name,
        model_card_data=SparseEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name="csr-mpnet-base trained on Natural Questions",
        ),
    )

    model._first_module().requires_grad_(False)  # print the proportion of parameters that are trainable
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Total parameters: {num_params}, Trainable parameters: {num_trainable_params}, Trainable ratio: {num_trainable_params / num_params:.2%}"
    )
    # 2a. Load the NQ dataset: https://huggingface.co/datasets/sentence-transformers/natural-questions
    logging.info("Read the Natural Questions training dataset")
    full_dataset = load_dataset("sentence-transformers/natural-questions", split="train").select(range(100_000))
    dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    logging.info(train_dataset)
    logging.info(eval_dataset)

    # 3. Define our training loss.
    loss = losses.CSRLoss(model=model, loss=losses.SparseMultipleNegativesRankingLoss(model=model))

    # 4. Define evaluator. This is useful to keep track of alongside the evaluation loss.
    evaluators = []
    for k_dim in [128, 256]:
        evaluators.append(evaluation.SparseNanoBEIREvaluator(["msmarco", "nfcorpus", "nq"], max_active_dims=k_dim))
    dev_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])

    # 5. Define the training arguments
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"csr-{short_model_name}-nq"
    training_args = SparseEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=4e-5,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoBEIR_mean_dot_ndcg@10",
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=309,
        save_strategy="steps",
        save_steps=309,
        logging_steps=10,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
    )

    # 6. Create the trainer & start training
    trainer = SparseEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # 7. Evaluate the final model, using the complete NanoBEIR dataset
    test_evaluator = evaluation.SparseNanoBEIREvaluator(show_progress_bar=True, batch_size=train_batch_size)
    test_evaluator(model)

    # 8. Save the final model
    final_output_dir = f"models/{run_name}/final"
    model.save_pretrained(final_output_dir)

    # 9. (Optional) save the model to the Hugging Face Hub!
    # It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
    try:
        model.push_to_hub(run_name)
    except Exception:
        logging.error(
            f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
            f"`huggingface-cli login`, followed by loading the model using `model = SparseEncoder({final_output_dir!r})` "
            f"and saving it using `model.push_to_hub('{run_name}')`."
        )


if __name__ == "__main__":
    main()
