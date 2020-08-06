"""
This example uses a simple bag-of-words (BoW) approach. A sentence is mapped
to a sparse vector with e.g. 25,000 dimensions. Optionally, you can also use tf-idf.

To make the model trainable, we add multiple dense layers to create a Deep Averaging Network (DAN).
"""
import torch
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
from sentence_transformers.models.tokenizer.WordTokenizer import ENGLISH_STOP_WORDS
import logging
from datetime import datetime

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the dataset
batch_size = 32
sts_reader = STSBenchmarkDataReader('../datasets/stsbenchmark')
model_save_path = 'output/training_tf-idf_word_embeddings-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



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
train_data = SentencesDataset(sts_reader.get_examples('sts-train.csv'), model=model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(sts_reader.get_examples('sts-dev.csv'))

# Configure the training
num_epochs = 10
warmup_steps = math.ceil(len(train_data) * num_epochs / batch_size * 0.1) #10% of train data for warm-up
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
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(sts_reader.get_examples("sts-test.csv"))

model.evaluate(evaluator)