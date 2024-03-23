from torch.utils.data import DataLoader
from sentence_transformers import losses, util, SentenceTransformer, LoggingHandler
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
from zipfile import ZipFile
import logging
import csv
import math
import os

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout

###### Read Datasets ######
sts_dataset_path = "datasets/stsbenchmark.tsv.gz"
qqp_dataset_path = "quora-IR-dataset"
batch_size = 64
num_epochs = 1
model_name = "bert-base-uncased"

model = SentenceTransformer(model_name)

# Check if the QQP dataset exists. If not, download and extract
if not os.path.exists(qqp_dataset_path):
    logging.info("Dataset not found. Download")
    zip_save_path = "quora-IR-dataset.zip"
    util.http_get(url="https://sbert.net/datasets/quora-IR-dataset.zip", path=zip_save_path)
    with ZipFile(zip_save_path, "r") as zipIn:
        zipIn.extractall(qqp_dataset_path)

positive_pairs = []

with open(os.path.join(qqp_dataset_path, "classification/train_pairs.tsv"), encoding="utf8") as fIn:
    reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        if row["is_duplicate"] == "1":
            positive_pairs.append(InputExample(texts=[row["question1"], row["question2"]]))

qqp_train_texts = list({text for example in positive_pairs for text in example.texts})

train_dataloader = DataLoader(positive_pairs, shuffle=True, batch_size=batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model)

###### Classification ######
# Given (quesiton1, question2), is this a duplicate or not?
# The evaluator will compute the embeddings for both questions and then compute
# a cosine similarity. If the similarity is above a threshold, we have a duplicate.
logging.info("Read QQP dev dataset")

dev_sentences1 = []
dev_sentences2 = []
dev_labels = []

with open(os.path.join(qqp_dataset_path, "classification/dev_pairs.tsv"), encoding="utf8") as fIn:
    reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        dev_sentences1.append(row["question1"])
        dev_sentences2.append(row["question2"])
        dev_labels.append(int(row["is_duplicate"]))

evaluator = BinaryClassificationEvaluator(dev_sentences1, dev_sentences2, dev_labels, train_texts=qqp_train_texts)

# Configure the training.
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the bi-encoder model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
)

logging.info("Read QQP test dataset")
test_sentences1 = []
test_sentences2 = []
test_labels = []

with open(os.path.join(qqp_dataset_path, "classification/test_pairs.tsv"), encoding="utf8") as fIn:
    reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
    for idx, row in enumerate(reader):
        test_sentences1.append(row["question1"])
        test_sentences2.append(row["question2"])
        test_labels.append(int(row["is_duplicate"]))
        if idx >= 10000:
            break

evaluator = BinaryClassificationEvaluator(
    test_sentences1, test_sentences2, test_labels, train_texts=qqp_train_texts, name="qqp-test"
)
model.evaluate(evaluator)
