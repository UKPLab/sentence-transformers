"""
This examples trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for the STSbenchmark from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.

Usage:
python training_nli.py

OR
python training_nli.py pretrained_transformer_model_name
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import sys
import os
import gzip
import csv

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout


# Check if dataset exists. If not, download and extract  it
sts_dataset_path = "datasets/stsbenchmark.tsv.gz"

if not os.path.exists(sts_dataset_path):
    util.http_get("https://sbert.net/datasets/stsbenchmark.tsv.gz", sts_dataset_path)


# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else "distilbert-base-uncased"

# Read the dataset
train_batch_size = 128
num_epochs = 4
model_save_path = (
    "output/training_stsbenchmark_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark train dataset")

train_samples = []
dev_samples = []
test_samples = []
with gzip.open(sts_dataset_path, "rt", encoding="utf8") as fIn:
    reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row["score"]) / 5.0  # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[row["sentence1"], row["sentence2"]], label=score)

        if row["split"] == "dev":
            dev_samples.append(inp_example)
        elif row["split"] == "test":
            test_samples.append(inp_example)
        else:
            train_samples.append(inp_example)


train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CoSENTLoss(model=model)
train_texts = list(set(text for inp_example in train_samples for text in inp_example.texts))


logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name="sts-dev", train_texts=train_texts)


# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
    output_path=model_save_path,
)


##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    test_samples, name="sts-test", train_texts=train_texts
)
test_evaluator(model, output_path=model_save_path)

"""
Batch size: 128

Ensemble via training:
CoSENT:
2024-02-06 21:08:50 - Cosine-Similarity :       Pearson: 0.8301 Spearman: 0.8428
2024-02-06 21:08:50 - Manhattan-Distance:       Pearson: 0.8430 Spearman: 0.8386
2024-02-06 21:08:50 - Euclidean-Distance:       Pearson: 0.8436 Spearman: 0.8394
2024-02-06 21:08:50 - Dot-Product-Similarity:   Pearson: 0.4730 Spearman: 0.4651
2024-02-06 21:08:50 - Surprise-Similarity:      Pearson: 0.3608 Spearman: 0.7497

2024-02-07 13:26:39 - Cosine-Similarity :       Pearson: 0.8295 Spearman: 0.8418
2024-02-07 13:26:39 - Manhattan-Distance:       Pearson: 0.8424 Spearman: 0.8380
2024-02-07 13:26:39 - Euclidean-Distance:       Pearson: 0.8430 Spearman: 0.8386
2024-02-07 13:26:39 - Dot-Product-Similarity:   Pearson: 0.4939 Spearman: 0.4859
2024-02-07 13:26:39 - Surprise-Similarity:      Pearson: 0.3591 Spearman: 0.7448
2024-02-07 13:26:39 - Surprise-Similarity-Dev:  Pearson: 0.7722 Spearman: 0.7635

Cosine:

MNRL:
2024-02-06 21:11:20 - Cosine-Similarity :       Pearson: 0.6984 Spearman: 0.6986
2024-02-06 21:11:20 - Manhattan-Distance:       Pearson: 0.7206 Spearman: 0.7144
2024-02-06 21:11:20 - Euclidean-Distance:       Pearson: 0.7211 Spearman: 0.7149
2024-02-06 21:11:20 - Dot-Product-Similarity:   Pearson: 0.4269 Spearman: 0.4124
2024-02-06 21:11:20 - Surprise-Similarity:      Pearson: 0.2740 Spearman: 0.5605

Ensemble via embeddings2:
CoSENT:
2024-02-06 21:15:09 - Cosine-Similarity :       Pearson: 0.8293 Spearman: 0.8417
2024-02-06 21:15:09 - Manhattan-Distance:       Pearson: 0.8426 Spearman: 0.8382
2024-02-06 21:15:09 - Euclidean-Distance:       Pearson: 0.8430 Spearman: 0.8387
2024-02-06 21:15:09 - Dot-Product-Similarity:   Pearson: 0.4765 Spearman: 0.4687
2024-02-06 21:15:09 - Surprise-Similarity:      Pearson: 0.3956 Spearman: 0.7212

Cosine:

MNRL:
2024-02-06 21:13:28 - Cosine-Similarity :       Pearson: 0.6958 Spearman: 0.6975
2024-02-06 21:13:28 - Manhattan-Distance:       Pearson: 0.7111 Spearman: 0.7076
2024-02-06 21:13:28 - Euclidean-Distance:       Pearson: 0.7120 Spearman: 0.7079
2024-02-06 21:13:28 - Dot-Product-Similarity:   Pearson: 0.4387 Spearman: 0.4257
2024-02-06 21:13:28 - Surprise-Similarity:      Pearson: 0.3486 Spearman: 0.5562
"""
