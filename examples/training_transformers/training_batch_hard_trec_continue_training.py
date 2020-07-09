"""
This script trains sentence transformers with a batch hard loss function.

The TREC dataset will be automatically downloaded and put in the datasets/ directory

Usual triplet loss takes 3 inputs: anchor, positive, negative and optimizes the network such that
the positive sentence is closer to the anchor than the negative sentence. However, a challenge here is
to select good triplets. If the negative sentence is selected randomly, the training objective is often
too easy and the network fails to learn good representations.

Batch hard triplet loss (https://arxiv.org/abs/1703.07737) creates triplets on the fly. It requires that the
data is labeled (e.g. labels A, B, C) and we assume that samples with the same label are similar:
A sent1; A sent2; B sent3; B sent4
...

In a batch, it checks for sent1 with label A what is the other sentence with label A that is the furthest (hard positive)
which sentence with another label is the closest (hard negative example). It then tries to optimize this, i.e.
all sentences with the same label should be close and sentences for different labels should be clearly seperated.
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
    directory="datasets/trec/",
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
train_batch_size = 32
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
    provide_positive=False, #For BatchHardTripletLoss, we must set provide_positive and provide_negative to False
    provide_negative=False,
)
train_dataloader = DataLoader(dataset_train, shuffle=True, batch_size=train_batch_size)

### Triplet losses ####################
### There are 3 triplet loss variants:
### - BatchHardTripletLoss
### - BatchHardSoftMarginTripletLoss
### - BatchSemiHardTripletLoss
#######################################

#train_loss = losses.BatchHardTripletLoss(sentence_embedder=model)
#train_loss = losses.BatchHardSoftMarginTripletLoss(sentence_embedder=model)
train_loss = losses.BatchSemiHardTripletLoss(sentence_embedder=model)

logging.info("Read TREC val dataset")
dataset_dev = SentenceLabelDataset(examples=val, model=model)
dev_dataloader = DataLoader(dataset_dev, shuffle=False, batch_size=train_batch_size)
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
model = SentenceTransformer(output_path)

dataset_test = SentenceLabelDataset(examples=test, model=model)
test_dataloader = DataLoader(dataset_test, shuffle=False, batch_size=train_batch_size)
test_evaluator = TripletEvaluator(test_dataloader)

logging.info("Evaluating model")
model.evaluate(test_evaluator)
