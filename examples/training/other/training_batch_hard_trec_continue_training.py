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
)
from torch.utils.data import DataLoader
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import TripletEvaluator
from datetime import datetime


import logging
import os
import urllib.request
import random
from collections import defaultdict

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
        guid=1
        for line in open(full_path, "rb"):
            # there is one non-ASCII byte: sisterBADBYTEcity; replaced with space
            label, _, text = line.replace(b"\xf0", b" ").strip().decode().partition(" ")
            
            # We extract the upper category (e.g. DESC from DESC:def)
            label, _, _ = label.partition(":")

            if label not in label_map:
                label_map[label] = len(label_map)

            label_id = label_map[label]
            guid += 1
            examples.append(InputExample(guid=guid, texts=[text], label=label_id))
        ret.append(examples)

    train_set, test_set = ret
    dev_set = None

    # Create a dev set from train set
    if validation_dataset_nb > 0:
        dev_set = train_set[-validation_dataset_nb:]
        train_set = train_set[:-validation_dataset_nb]

    # For dev & test set, we return triplets (anchor, positive, negative)
    random.seed(42) #Fix seed, so that we always get the same triplets
    dev_triplets = triplets_from_labeled_dataset(dev_set)
    test_triplets = triplets_from_labeled_dataset(test_set)

    return train_set, dev_triplets, test_triplets


def triplets_from_labeled_dataset(input_examples):
    # Create triplets for a [(label, sentence), (label, sentence)...] dataset
    # by using each example as an anchor and selecting randomly a
    # positive instance with the same label and a negative instance with a different label
    triplets = []
    label2sentence = defaultdict(list)
    for inp_example in input_examples:
        label2sentence[inp_example.label].append(inp_example)

    for inp_example in input_examples:
        anchor = inp_example

        if len(label2sentence[inp_example.label]) < 2: #We need at least 2 examples per label to create a triplet
            continue

        positive = None
        while positive is None or positive.guid == anchor.guid:
            positive = random.choice(label2sentence[inp_example.label])

        negative = None
        while negative is None or negative.label == anchor.label:
            negative = random.choice(input_examples)

        triplets.append(InputExample(texts=[anchor.texts[0], positive.texts[0], negative.texts[0]]))

    return triplets


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = 'distilbert-base-nli-stsb-mean-tokens'

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
train_set, dev_set, test_set = trec_dataset()



# Load pretrained model
model = SentenceTransformer(model_name)

logging.info("Read TREC train dataset")
train_dataset = SentenceLabelDataset(
    examples=train_set,
    model=model,
    provide_positive=False, #For BatchHardTripletLoss, we must set provide_positive and provide_negative to False
    provide_negative=False,
)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)

### Triplet losses ####################
### There are 4 triplet loss variants:
### - BatchHardTripletLoss
### - BatchHardSoftMarginTripletLoss
### - BatchSemiHardTripletLoss
### - BatchAllTripletLoss
#######################################

train_loss = losses.BatchAllTripletLoss(model=model)
#train_loss = losses.BatchHardTripletLoss(sentence_embedder=model)
#train_loss = losses.BatchHardSoftMarginTripletLoss(sentence_embedder=model)
#train_loss = losses.BatchSemiHardTripletLoss(sentence_embedder=model)


logging.info("Read TREC val dataset")
dev_evaluator = TripletEvaluator.from_input_examples(dev_set, name='dev')

logging.info("Performance before fine-tuning:")
dev_evaluator(model)

warmup_steps = int(
    len(train_dataset) * num_epochs / train_batch_size * 0.1
)  # 10% of train data

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
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

logging.info("Evaluating model on test set")
test_evaluator = TripletEvaluator.from_input_examples(test_set, name='test')
model.evaluate(test_evaluator)
