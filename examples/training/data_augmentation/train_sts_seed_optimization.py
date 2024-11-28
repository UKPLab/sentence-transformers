"""
This script is identical to examples/training/sts/training_stsbenchmark.py with seed optimization.
We apply early stopping and evaluate the models over the dev set, to find out the best performing seeds.

For more details refer to -
Fine-Tuning Pretrained Language Models:
Weight Initializations, Data Orders, and Early Stopping by Dodge et al. 2020
https://arxiv.org/pdf/2002.06305.pdf

Why Seed Optimization?
Dodge et al. (2020) show a high dependence on the random seed for transformer based models like BERT,
as it converges to different minima that generalize differently to unseen data. This is especially the
case for small training datasets.

Citation: https://arxiv.org/abs/2010.08240

Usage:
python train_sts_seed_optimization.py

OR
python train_sts_seed_optimization.py pretrained_transformer_model_name seed_count stop_after

python train_sts_seed_optimization.py bert-base-uncased 10 0.3
"""

import logging
import math
import random
import sys

import numpy as np
import torch
from datasets import load_dataset

from sentence_transformers import LoggingHandler, SentenceTransformer, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)


# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased"
seed_count = int(sys.argv[2]) if len(sys.argv) > 2 else 10
stop_after = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3

logging.info(f"Train and Evaluate: {seed_count} Random Seeds")

for seed in range(seed_count):
    # Setting seed for all random initializations
    logging.info(f"##### Seed {seed} #####")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Read the dataset
    train_batch_size = 16
    num_epochs = 1
    model_save_path = "output/bi-encoder/training_stsbenchmark_" + model_name + "/seed-" + str(seed)

    # Use Hugging Face/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # 2. Load the STSB dataset: https://huggingface.co/datasets/sentence-transformers/stsb
    train_dataset = load_dataset("sentence-transformers/stsb", split="train")
    eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
    test_dataset = load_dataset("sentence-transformers/stsb", split="test")
    logging.info(train_dataset)

    train_loss = losses.CosineSimilarityLoss(model=model)

    # 4. Define an evaluator for use during training.
    dev_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=eval_dataset["sentence1"],
        sentences2=eval_dataset["sentence2"],
        scores=eval_dataset["score"],
        main_similarity=SimilarityFunction.COSINE,
        name="sts-dev",
    )

    # Stopping and Evaluating after 30% of training data (less than 1 epoch)
    # We find from (Dodge et al.) that 20-30% is often ideal for convergence of random seed
    steps_per_epoch = math.ceil(len(train_dataset) * stop_after)

    logging.info(f"Early-stopping: {int(stop_after * 100)}% of the training-data")

    # 5. Define the training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=model_save_path,
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        warmup_ratio=0.1,
        max_steps=steps_per_epoch,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        # Optional tracking/debugging parameters:
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=1000,
        run_name="sts",  # Will be used in W&B if `wandb` is installed
    )

    # 6. Create the trainer & start training
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
    )
    trainer.train()
