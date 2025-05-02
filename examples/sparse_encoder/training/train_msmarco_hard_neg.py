import logging

import torch
from datasets import load_dataset, load_from_disk

from sentence_transformers import SparseEncoder, SparseEncoderTrainer, SparseEncoderTrainingArguments, losses
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator


def main():
    # Set the log level to INFO to get more information
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    train_batch_size = 124
    num_epochs = 35  # 1 epochs is 532K row
    num_hard_negatives = 8
    lambda_query = 5e-4
    lambda_corpus = 3e-4
    learning_rate = 2e-5

    # 1. Define our SparseEncoder model
    # Set the seed so the new classifier weights are identical in subsequent runs
    torch.manual_seed(12)
    model_name = "sparse-embedding/splade-distilbert-base-uncased-init"
    model = SparseEncoder(model_name)

    # 2. Load the MS MARCO dataset: https://huggingface.co/datasets/sentence-transformers/msmarco
    logging.info("Read train dataset")
    try:
        train_dataset = load_from_disk(f"datasets/ms-marco-train-hard-negatives-{num_hard_negatives}")
        eval_dataset = load_from_disk(f"datasets/ms-marco-eval-hard-negatives-{num_hard_negatives}")
    except FileNotFoundError:
        logging.info("The dataset has not been fully stored as texts on disk yet. We will do this now.")
        passage_dataset = load_dataset("sentence-transformers/msmarco-corpus", "passage", split="train")
        pid_to_passage = dict(zip(passage_dataset["pid"], passage_dataset["text"]))
        query_dataset = load_dataset("sentence-transformers/msmarco-corpus", "query", split="train")
        qid_to_query = dict(zip(query_dataset["qid"], query_dataset["text"]))
        dataset = load_dataset(
            "sparse-embedding/msmarco-hard-negatives-cross-encoder-ms-marco-MiniLM-L-6-v2-scores",
            split="train",
        )

        def id_to_text_map(batch):
            negative_ids = batch["negative_ids"]  # (batch_size, list of negative ids)
            negative_scores = batch["negative_scores"]  # (batch_size, list of negative scores)

            mapped_negatives = []
            mapped_negatives_scores = torch.empty((len(negative_ids), num_hard_negatives))

            for neg_ids, neg_scores in zip(negative_ids, negative_scores):
                neg_scores = torch.tensor(neg_scores)
                # Get indices of sorted negatives (highest score first)
                sorted_indices = torch.argsort(neg_scores, descending=True)[:num_hard_negatives]

                mapped_neg_texts = [pid_to_passage.get(pid, "") for pid in [neg_ids[i] for i in sorted_indices]]

                mapped_negatives.append(mapped_neg_texts)
                mapped_negatives_scores[len(mapped_negatives) - 1] = neg_scores[sorted_indices]

            return {
                "query": [qid_to_query.get(qid, "") for qid in batch["query_id"]],
                "positive": [pid_to_passage.get(pid, "") for pid in batch["positive_id"]],
                **{
                    f"negative_{i + 1}": [neg_list[i] for neg_list in mapped_negatives]
                    for i in range(num_hard_negatives)
                },
                "label": torch.tensor(batch["positive_score"]).unsqueeze(1) - mapped_negatives_scores,
            }

        dataset = dataset.map(id_to_text_map, batched=True, remove_columns=dataset.column_names)
        dataset = dataset.train_test_split(test_size=10_000)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        train_dataset.save_to_disk(f"datasets/ms-marco-train-hard-negatives-{num_hard_negatives}")
        eval_dataset.save_to_disk(f"datasets/ms-marco-eval-hard-negatives-{num_hard_negatives}")

        logging.info(
            "The dataset has now been stored as texts on disk. The script will now stop to ensure that memory is freed. "
            "Please restart the script to start training."
        )
        quit()

    logging.info(train_dataset)

    # 3. Define our training loss
    loss = losses.SpladeLoss(
        model, losses.SparseMarginMSELoss(model), lambda_query=lambda_query, lambda_corpus=lambda_corpus
    )

    # 4. Define the evaluator. We use the SparseNanoBEIREvaluator, which is a light-weight evaluator for English
    evaluator = SparseNanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"], batch_size=train_batch_size)

    # 5. Define the training arguments
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"{short_model_name}-msmarco-hard-negatives-{num_hard_negatives}-bs-{train_batch_size}-lq-{loss.lambda_query}-lc-{loss.lambda_corpus}"
    args = SparseEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=learning_rate,
        warmup_steps=6000,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        # Optional tracking/debugging parameters:
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=1,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        seed=12,
        dataloader_num_workers=4,
    )

    # 6. Create the trainer & start training
    trainer = SparseEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    # 7. Evaluate the final model, useful to include these in the model card
    evaluator(model)

    # 8. Save the final model
    final_output_dir = f"models/{run_name}/final"
    model.save_pretrained(final_output_dir)


if __name__ == "__main__":
    main()
