from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Training parameters
model_name = 'bert-base-uncased'
train_batch_size = 8
num_epochs = 1
max_seq_length = 75

# Save path to store our model
model_save_path = 'output/training_stsb_tsdae-{}-{}-{}'.format(model_name, train_batch_size, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

# Check if dataset exsist. If not, download and extract  it
sts_dataset_path = 'data/stsbenchmark.tsv.gz'

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

# Defining our sentence transformer model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# We use 1 Million sentences from Wikipedia to train our model
wikipedia_dataset_path = 'data/wiki1m_for_simcse.txt'
if not os.path.exists(wikipedia_dataset_path):
    util.http_get('https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt', wikipedia_dataset_path)

# train_samples is a list of InputExample objects where we pass the same sentence twice to texts, i.e. texts=[sent, sent]
train_sentences = []
with open(wikipedia_dataset_path, 'r', encoding='utf8') as fIn:
    for line in fIn:
        line = line.strip()
        if len(line) >= 10:
            train_sentences.append(line)

# Read STSbenchmark dataset and use it as development set
logging.info("Read STSbenchmark dev dataset")
dev_samples = []
test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1

        if row['split'] == 'dev':
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
        elif row['split'] == 'test':
            test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')

# We train our model using the MultipleNegativesRankingLoss
train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)
train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=True)


evaluation_steps = 1000
logging.info("Training sentences: {}".format(len(train_sentences)))
logging.info("Performance before training")
dev_evaluator(model)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=evaluation_steps,
          output_path=model_save_path,
          weight_decay=0,
          warmup_steps=100,
          optimizer_params={'lr': 3e-5},
          use_amp=True         #Set to True, if your GPU supports FP16 cores
          )

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################


model = SentenceTransformer(model_save_path)
test_evaluator(model, output_path=model_save_path)
