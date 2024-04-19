"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with MatryoshkaLoss using MultipleNegativesRankingLoss. This trains a model at output dimensions [768, 512, 256, 128, 64].
Entailments are positive pairs and the contradiction on AllNLI dataset is added as a hard negative.
At every 10% training steps, the model is evaluated on the STS benchmark dataset at the different output dimensions.

The difference between this script and matryoshka_nli.py is that this script uses a reduced dimensionality of the base
model by adding a Dense layer with `reduced_dim=256` output dimensions. This might be useful when your desired output
dimensionality is lower than the base model's default output dimensionality.

Usage:
python matryoshka_nli_reduced_dim.py

OR
python matryoshka_nli_reduced_dim.py pretrained_transformer_model_name
"""

import math
from datasets import load_dataset
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator, SimilarityFunction
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import random

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout

model_name = sys.argv[1] if len(sys.argv) > 1 else "distilroberta-base"
train_batch_size = 128  # The larger you select this, the better the results (usually). But it requires more GPU memory
max_seq_length = 75
num_epochs = 1
reduced_dim = 256
matryoshka_dims = [256, 128, 64, 32, 16]

# Save path of the model
model_save_path = (
    "output/matryoshka_nli_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)


# Here we define our SentenceTransformer model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean")
dense = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=reduced_dim)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense])

# Check if dataset exists. If not, download and extract  it
nli_dataset_path = "data/AllNLI.tsv.gz"

if not os.path.exists(nli_dataset_path):
    util.http_get("https://sbert.net/datasets/AllNLI.tsv.gz", nli_dataset_path)

# Read the AllNLI.tsv.gz file and create the training dataset
logging.info("Read AllNLI train dataset")


def add_to_samples(sent1, sent2, label):
    if sent1 not in train_data:
        train_data[sent1] = {"contradiction": set(), "entailment": set(), "neutral": set()}
    train_data[sent1][label].add(sent2)


train_data = {}
with gzip.open(nli_dataset_path, "rt", encoding="utf8") as fIn:
    reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        if row["split"] == "train":
            sent1 = row["sentence1"].strip()
            sent2 = row["sentence2"].strip()

            add_to_samples(sent1, sent2, row["label"])
            add_to_samples(sent2, sent1, row["label"])  # Also add the opposite


train_samples = []
for sent1, others in train_data.items():
    if len(others["entailment"]) > 0 and len(others["contradiction"]) > 0:
        train_samples.append(
            InputExample(
                texts=[sent1, random.choice(list(others["entailment"])), random.choice(list(others["contradiction"]))]
            )
        )
        train_samples.append(
            InputExample(
                texts=[random.choice(list(others["entailment"])), sent1, random.choice(list(others["contradiction"]))]
            )
        )

logging.info("Train samples: {}".format(len(train_samples)))


# Special data loader that avoid duplicates within a batch
train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)


# Our training loss
train_loss = losses.MultipleNegativesRankingLoss(model)
train_loss = losses.MatryoshkaLoss(model, train_loss, matryoshka_dims=matryoshka_dims)

stsb_dev = load_dataset("mteb/stsbenchmark-sts", split="validation")
evaluators = []
for dim in matryoshka_dims:
    evaluators.append(
        EmbeddingSimilarityEvaluator(
            stsb_dev["sentence1"],
            stsb_dev["sentence2"],
            [score / 5 for score in stsb_dev["score"]],
            main_similarity=SimilarityFunction.COSINE,
            name=f"sts-dev-{dim}",
            truncate_dim=dim,
        )
    )
dev_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[0])

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    epochs=num_epochs,
    evaluation_steps=int(len(train_dataloader) * 0.1),
    warmup_steps=warmup_steps,
    output_path=model_save_path,
    use_amp=False,  # Set to True, if your GPU supports FP16 operations
)


##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################


model = SentenceTransformer(model_save_path)
stsb_test = load_dataset("mteb/stsbenchmark-sts", split="test")
evaluators = []
for dim in matryoshka_dims:
    evaluators.append(
        EmbeddingSimilarityEvaluator(
            stsb_test["sentence1"],
            stsb_test["sentence2"],
            [score / 5 for score in stsb_test["score"]],
            main_similarity=SimilarityFunction.COSINE,
            name=f"sts-test-{dim}",
            truncate_dim=dim,
        )
    )
test_evaluator = SequentialEvaluator(evaluators)
test_evaluator(model, output_path=model_save_path)


# Optionally, save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-nli-matryoshka-{reduced_dim}")
except Exception:
    logging.error(
        "Error uploading model to the Hugging Face Hub. To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({model_save_path!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-nli-matryoshka-{reduced_dim}')`."
    )
