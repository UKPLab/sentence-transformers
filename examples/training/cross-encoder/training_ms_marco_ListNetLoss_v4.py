import logging
import traceback

from datasets import load_dataset

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation.CENanoBEIREvaluator import (
    CENanoBEIREvaluator,
)
from sentence_transformers.cross_encoder.losses import ListNetLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import (
    CrossEncoderTrainingArguments,
)


def main():
    model_name = "microsoft/MiniLM-L12-H384-uncased"

    # Set the log level to INFO to get more information
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    # The batch size is lower because we have to process multiple documents per query
    # This means that the batch size is effectively multiplied by the number of max_docs
    train_batch_size = 6
    eval_batch_size = 16
    num_epochs = 3
    max_docs = 10
    pad_value = -1
    loss_name = "listnet"
    num_labels = 1

    # 1. Define our CrossEncoder model
    model = CrossEncoder(model_name, num_labels=num_labels)
    print("Model max length:", model.max_length)
    print("Model num labels:", model.num_labels)

    # 2. Load the MS MARCO dataset: https://huggingface.co/datasets/microsoft/ms_marco
    logging.info("Read train dataset")
    dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")

    def listwise_mapper(batch, max_docs: int = 10, pad_value: int = -1):
        processed_queries = []
        processed_docs = []
        processed_labels = []

        for query, passages_info in zip(batch["query"], batch["passages"]):
            # Extract passages and labels
            passages = passages_info["passage_text"]
            labels = passages_info["is_selected"]

            # Pair passages with labels and sort descending by label (positives first)
            paired = sorted(zip(passages, labels), key=lambda x: x[1], reverse=True)

            # Separate back to passages and labels
            sorted_passages, sorted_labels = zip(*paired) if paired else ([], [])

            # Filter queries without any positive labels
            if max(sorted_labels) < 1.0:
                continue

            # Truncate to max_docs
            truncated_passages = list(sorted_passages[:max_docs])
            truncated_labels = list(sorted_labels[:max_docs])

            # Pad if needed
            pad_count = max_docs - len(truncated_passages)
            processed_docs.append(truncated_passages + [""] * pad_count)
            processed_labels.append(truncated_labels + [pad_value] * pad_count)
            processed_queries.append(query)

        return {
            "query": processed_queries,
            "docs": processed_docs,
            "labels": processed_labels,
        }

    dataset = dataset.map(
        lambda batch: listwise_mapper(batch=batch, max_docs=max_docs, pad_value=pad_value),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Processing listwise samples",
    )

    dataset = dataset.train_test_split(test_size=10_000)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    logging.info(train_dataset)

    # 3. Define our training loss
    loss = ListNetLoss(model, pad_value=pad_value)

    # 4. Define the evaluator. We use the CENanoBEIREvaluator, which is a light-weight evaluator for English reranking
    evaluator = CENanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"], batch_size=eval_batch_size)
    evaluator(model)

    # 5. Define the training arguments
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"reranker-msmarco-v1.1-{short_model_name}-{loss_name}"
    args = CrossEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoBEIR_R100_mean_ndcg@10",
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=4_000,
        save_strategy="steps",
        save_steps=4_000,
        save_total_limit=2,
        logging_steps=1_000,
        logging_first_step=True,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        seed=12,
    )

    # 6. Create the trainer & start training
    trainer = CrossEncoderTrainer(
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

    # 9. (Optional) save the model to the Hugging Face Hub!
    # It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
    try:
        model.push_to_hub(run_name)
    except Exception:
        logging.error(
            f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
            f"`huggingface-cli login`, followed by loading the model using `model = CrossEncoder({final_output_dir!r})` "
            f"and saving it using `model.push_to_hub('{run_name}')`."
        )


if __name__ == "__main__":
    main()
