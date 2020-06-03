"""
This script trains sentence transformers with a batch hard loss function.

The TREC dataset will be automatically downloaded and put in the datasets/ directory
"""

from sentence_transformers import (
    SentenceTransformer,
    SentenceLabelDataset,
    LoggingHandler,
    losses,
    models,
)
from torch.utils.data import DataLoader
from sentence_transformers.readers import TripletReader, InputExample
from sentence_transformers.evaluation import TripletEvaluator
from datetime import datetime

import csv
import logging
import os
import urllib.request

# Inspired from torchnlp
def trec_dataset(
    directory="sentence-transformers/datasets/trec/",
    train_filename="train_5500.label",
    test_filename="TREC_10.label",
    validation_dataset_nb=500,
    urls=[
        "http://cogcomp.org/Data/QA/QC/train_5500.label",
        "http://cogcomp.org/Data/QA/QC/TREC_10.label",
    ],
):

    os.makedirs(directory, exist_ok=True)

    ret = []
    for url, filename in zip(urls, [train_filename, test_filename]):
        full_path = os.path.join(directory, filename)
        urllib.request.urlretrieve(url, filename=full_path)

        examples = []
        label_map = {}
        guid = 0
        for line in open(full_path, "rb"):
            # there is one non-ASCII byte: sisterBADBYTEcity; replaced with space
            label, _, text = line.replace(b"\xf0", b" ").strip().decode().partition(" ")
            
            # We extract the upper category (e.g. DESC from DESC:def)
            label, _, _ = label.partition(":")

            if label not in label_map:
                label_map[label] = len(label_map)

            guid += 1
            label_id = label_map[label]
            examples.append(InputExample(guid=guid, texts=[text], label=label_id))
        ret.append(examples)

    # Validation dataset:
    # It doesn't exist in the original dataset,
    # so we create one by splitting the train data
    # Ret[0] is train
    # Ret[1] is test
    # Ret[2] is val
    if validation_dataset_nb > 0:
        ret.append(ret[0][-validation_dataset_nb:])
        ret[0] = ret[0][:-validation_dataset_nb]

    return ret


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = 'bert-base-nli-stsb-mean-tokens'

### Create a torch.DataLoader that passes training batch instances to our model
train_batch_size = 16
output_path = (
    "output/finetune-batch-hard-trec-"
    + model_name
    + "-"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
num_epochs = 1

logging.info("Loading TREC dataset")
train, test, val = trec_dataset()

# Load pretrained model
model = SentenceTransformer(model_name)

logging.info("Read TREC train dataset")
dataset_train = SentenceLabelDataset(
    examples=train,
    model=model,
    provide_positive=False,
    provide_negative=False,
)
train_dataloader = DataLoader(dataset_train, shuffle=True, batch_size=train_batch_size)
train_loss = losses.BatchHardTripletLoss(sentence_embedder=model)

logging.info("Read TREC val dataset")
dataset_dev = SentenceLabelDataset(examples=val, model=model)
dev_dataloader = DataLoader(dataset_dev, shuffle=True, batch_size=train_batch_size)
evaluator = TripletEvaluator(dev_dataloader)

warmup_steps = int(
    len(train_dataloader) * num_epochs / train_batch_size * 0.1
)  # 10% of train data

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
    output_path=output_path,
)

##############################################################################
#
# Load the stored model and evaluate its performance on TREC dataset
#
##############################################################################

logging.info("Read TREC test dataset")
dataset_test = SentenceLabelDataset(examples=val, model=model)
test_dataloader = DataLoader(dataset_test, shuffle=True, batch_size=train_batch_size)
test_evaluator = TripletEvaluator(test_dataloader)

logging.info("Evaluating model")
model = SentenceTransformer(output_path)
model.evaluate(test_evaluator)
