"""
This examples trains a CrossEncoder for the NLI task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it learns to predict the labels: "contradiction": 0, "entailment": 1, "neutral": 2.

It does NOT produce a sentence embedding and does NOT work for individual sentences.

Usage:
python training_nli.py
"""

import logging
from datetime import datetime

from datasets import load_dataset

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEClassificationEvaluator
from sentence_transformers.cross_encoder.losses.CrossEntropyLoss import CrossEntropyLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

train_batch_size = 64
num_epochs = 1
output_dir = "output/training_ce_allnli-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 1. Define our CrossEncoder model. We use distilroberta-base as basis and setup it up to predict 3 labels
# You can also use other base models, like bert-base-uncased, microsoft/mpnet-base, etc.
model = CrossEncoder("distilroberta-base", num_labels=3)

# 2. Load the AllNLI dataset: https://huggingface.co/datasets/sentence-transformers/all-nli
# We'll start with 10k training samples, but you can increase this to get a stronger model
logging.info("Read AllNLI train dataset")
train_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="train").select(range(10000))
eval_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="dev").select(range(1000))
test_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="test")
logging.info(train_dataset)

# We might want to remap labels from the dataset, you can do that like so:
mapping = {0: 1, 1: 2, 2: 0}
eval_dataset = eval_dataset.map(lambda x: {"label": mapping[x["label"]]})
test_dataset = test_dataset.map(lambda x: {"label": mapping[x["label"]]})

# 3. Define our training loss:
loss = CrossEntropyLoss(model)

# During training, we use CEClassificationEvaluator to measure the performance on the dev set
dev_cls_evaluator = CEClassificationEvaluator(
    list(zip(eval_dataset["premise"], eval_dataset["hypothesis"])),
    eval_dataset["label"],
    name="AllNLI-dev",
)
dev_cls_evaluator(model)

# 5. Define the training arguments
args = CrossEncoderTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    run_name="ce-nli-v1",  # Will be used in W&B if `wandb` is installed
)

# 6. Create the trainer & start training
trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_cls_evaluator,
)
trainer.train()

# 7. Evaluate the final model on test dataset
test_cls_evaluator = CEClassificationEvaluator(
    list(zip(test_dataset["premise"], test_dataset["hypothesis"])),
    test_dataset["label"],
    name="AllNLI-test",
)
test_cls_evaluator(model)

# 8. Evaluate the final model and save it
final_output_dir = f"{output_dir}/final"
model.save_pretrained(final_output_dir)
