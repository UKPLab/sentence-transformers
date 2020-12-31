"""
This example uses a simple bag-of-words (BoW) approach. A sentence is mapped
to a sparse vector with e.g. 25,000 dimensions. Optionally, you can also use tf-idf.

To make the model trainable, we add multiple dense layers to create a Deep Averaging Network (DAN).
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses, util
from sentence_transformers import LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
from sentence_transformers.models.tokenizer.WordTokenizer import ENGLISH_STOP_WORDS
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
model_save_path = 'output/training_tf-idf_word_embeddings-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



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

##### Construction of the SentenceTransformer Model #####

# Wikipedia document frequency for words
wiki_doc_freq = 'wikipedia_doc_frequencies.txt'
if not os.path.exists(wiki_doc_freq):
    util.http_get('https://public.ukp.informatik.tu-darmstadt.de/reimers/embeddings/wikipedia_doc_frequencies.txt', wiki_doc_freq)

# Create the vocab for the BoW model
stop_words = ENGLISH_STOP_WORDS
max_vocab_size = 25000 #This is also the size of the BoW sentence vector.


#Read the most common max_vocab_size words. Skip stop-words
vocab = set()
weights = {}
lines = open('wikipedia_doc_frequencies.txt', encoding='utf8').readlines()
num_docs = int(lines[0])
for line in lines[1:]:
    word, freq = line.lower().strip().split("\t")
    if word in stop_words:
        continue

    vocab.add(word)
    weights[word] = math.log(num_docs/int(freq))

    if len(vocab) >= max_vocab_size:
        break

##### Construction of the SentenceTransformer Model #####

#Create the BoW model. Because we set word_weights to the IDF values and cumulative_term_frequency=True, we
#get tf-idf vectors. Set word_weights to an empty dict and cumulative_term_frequency=False to get a 1-hot sentence encoding
bow = models.BoW(vocab=vocab, word_weights=weights, cumulative_term_frequency=True)

# Add two trainable feed-forward networks (DAN) with max_vocab_size -> 768 -> 512 dimensions.
sent_embeddings_dimension = max_vocab_size
dan1 = models.Dense(in_features=sent_embeddings_dimension, out_features=768)
dan2 = models.Dense(in_features=768, out_features=512)

model = SentenceTransformer(modules=[bow, dan1, dan2])


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