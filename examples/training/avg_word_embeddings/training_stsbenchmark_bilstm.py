"""
This example runs a BiLSTM after the word embedding lookup. The output of the BiLSTM is than pooled,
for example with max-pooling (which gives a system like InferSent) or with mean-pooling.

Note, you can also pass BERT embeddings to the BiLSTM.
"""
import torch
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses, util
from sentence_transformers import LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
import logging
from datetime import datetime
import os
import csv
import gzip

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the dataset
batch_size = 32
model_save_path = 'output/training_stsbenchmark_bilstm-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


#Check if dataset exsist. If not, download and extract  it
sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

logging.info("Read STSbenchmark train dataset")

train_samples = []
dev_samples = []
test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

        if row['split'] == 'dev':
            dev_samples.append(inp_example)
        elif row['split'] == 'test':
            test_samples.append(inp_example)
        else:
            train_samples.append(inp_example)

# Map tokens to traditional word embeddings like GloVe
word_embedding_model = models.WordEmbeddings.from_text_file('glove.6B.300d.txt.gz')

lstm = models.LSTM(word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(), hidden_dim=1024)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(lstm.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=False,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=True)


model = SentenceTransformer(modules=[word_embedding_model, lstm, pooling_model])


# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark train dataset")
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

# Configure the training
num_epochs = 10
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )



##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
model.evaluate(evaluator)