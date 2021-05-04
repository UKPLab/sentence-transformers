"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with MultipleNegativesRankingLoss. Entailnments are poisitive pairs and the contradiction on AllNLI dataset is added as a hard negative.
At every 10% training steps, the model is evaluated on the STS benchmark dataset

Usage:
python training_nli_v2.py

OR
python training_nli_v2.py pretrained_transformer_model_name
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import random

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

model_name = sys.argv[1] if len(sys.argv) > 1 else 'distilroberta-base'
train_batch_size = 128          #The larger you select this, the better the results (usually). But it requires more GPU memory
max_seq_length = 75
num_epochs = 1

# Save path of the model
model_save_path = 'output/training_nli_v2_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Here we define our SentenceTransformer model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

#Check if dataset exsist. If not, download and extract  it
nli_dataset_path = 'data/AllNLI.tsv.gz'
sts_dataset_path = 'data/stsbenchmark.tsv.gz'

if not os.path.exists(nli_dataset_path):
    util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)


# Read the AllNLI.tsv.gz file and create the training dataset
logging.info("Read AllNLI train dataset")

def add_to_samples(sent1, sent2, label):
    if sent1 not in train_data:
        train_data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
    train_data[sent1][label].add(sent2)


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')

label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
train_data = {}
with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'train':
            sent1 = row['sentence1'].strip()
            sent2 = row['sentence2'].strip()

            add_to_samples(sent1, sent2, row['label'])
            add_to_samples(sent2, sent1, row['label'])  #Also add the opposite



train_samples = []
for sent1, others in train_data.items():
    if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
        train_samples.append(InputExample(texts=[sent1, random.choice(list(others['entailment'])), random.choice(list(others['contradiction']))]))
        train_samples.append(InputExample(texts=[random.choice(list(others['entailment'])), sent1, random.choice(list(others['contradiction']))]))

logging.info("Train samples: {}".format(len(train_samples)))


class NoDuplicatesSampler:
    def __init__(self, train_examples, batch_size):
        self.train_examples = train_examples
        self.batch_size = batch_size
        self.data_idx = list(range(len(train_examples)))
        self.data_pointer = 0
        random.shuffle(self.data_idx)

    def __iter__(self):
        for _ in range(self.__len__()):
            batch_idx = set()
            texts_in_batch = set()

            while len(batch_idx) < self.batch_size:
                idx = self.data_idx[self.data_pointer]
                example = self.train_examples[idx]

                valid_example = True
                for text in example.texts:
                    if text.strip().lower() in texts_in_batch:
                        valid_example = False
                        break

                if valid_example:
                    batch_idx.add(idx)
                    for text in example.texts:
                        texts_in_batch.add(text.strip().lower())

                self.data_pointer += 1
                if self.data_pointer >= len(self.data_idx):
                    self.data_pointer = 0
                    random.shuffle(self.data_idx)

            yield list(batch_idx)

    def __len__(self):
        return math.ceil(len(self.train_examples) / self.batch_size)

# Standard data loader
#train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)

# Special data loader that avoid duplicates within a batch
train_dataloader = DataLoader(train_samples, batch_sampler=NoDuplicatesSampler(train_samples, batch_size=train_batch_size))

# Our training loss
train_loss = losses.MultipleNegativesRankingLoss(model)




#Read STSbenchmark dataset and use it as development set
logging.info("Read STSbenchmark dev dataset")
dev_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'dev':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=int(len(train_dataloader)*0.1),
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=False          #Set to True, if your GPU supports FP16 operations
          )



##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'test':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')
test_evaluator(model, output_path=model_save_path)
