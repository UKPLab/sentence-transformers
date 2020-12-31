"""
This example weights word embeddings (like GloVe) with IDF weights. The IDF weights can for example be computed on Wikipedia.

If 'glove.6B.300d.txt.gz' does not exist, it tries to download it from our server.

See https://public.ukp.informatik.tu-darmstadt.de/reimers/embeddings/ for available word embeddings files

You can get term-document frequencies from here:
https://public.ukp.informatik.tu-darmstadt.de/reimers/embeddings/wikipedia_doc_frequencies.txt
"""
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
model_save_path = 'output/training_tf-idf_word_embeddings-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Wikipedia document frequency for words
wiki_doc_freq = 'wikipedia_doc_frequencies.txt'
if not os.path.exists(wiki_doc_freq):
    util.http_get('https://public.ukp.informatik.tu-darmstadt.de/reimers/embeddings/wikipedia_doc_frequencies.txt', wiki_doc_freq)

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

# Map tokens to traditional word embeddings like GloVe
word_embedding_model = models.WordEmbeddings.from_text_file('glove.6B.300d.txt.gz')

# Weight word embeddings using Inverse-Document-Frequency (IDF) values.
# For each word in the vocab ob the tokenizer, we must specify a weight value.
# The word embedding is then multiplied by this value
vocab = word_embedding_model.tokenizer.get_vocab()
word_weights = {}
lines = open(wiki_doc_freq, encoding='utf8').readlines()
num_docs = int(lines[0])
for line in lines[1:]:
    word, freq = line.strip().split("\t")
    word_weights[word] = math.log(num_docs/int(freq))

# Words in the vocab that are not in the doc_frequencies file get a frequency of 1
unknown_word_weight = math.log(num_docs/1)

# Initialize the WordWeights model. This model must be between the WordEmbeddings and the Pooling model
word_weights = models.WordWeights(vocab=vocab, word_weights=word_weights, unknown_word_weight=unknown_word_weight)


# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

# Add two trainable feed-forward networks (DAN)
sent_embeddings_dimension = pooling_model.get_sentence_embedding_dimension()
dan1 = models.Dense(in_features=sent_embeddings_dimension, out_features=sent_embeddings_dimension)
dan2 = models.Dense(in_features=sent_embeddings_dimension, out_features=sent_embeddings_dimension)

model = SentenceTransformer(modules=[word_embedding_model, word_weights, pooling_model, dan1, dan2])


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