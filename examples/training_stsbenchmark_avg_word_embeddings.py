"""
This example uses average word embeddings (for example from GloVe). It adds two fully-connected feed-forward layers (dense layers) to create a Deep Averaging Network (DAN).

If 'glove.6B.300d.txt.gz' does not exist, it tries to download it from our server.

See https://public.ukp.informatik.tu-darmstadt.de/reimers/embeddings/
for available word embeddings files
"""
import torch
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
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
sts_reader = STSDataReader('datasets/stsbenchmark')
model_save_path = 'output/training_stsbenchmark_avg_word_embeddings-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



# Map tokens to traditional word embeddings like GloVe
word_embedding_model = models.WordEmbeddings.from_text_file('glove.6B.300d.txt.gz')

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

# Add two trainable feed-forward networks (DAN)
sent_embeddings_dimension = pooling_model.get_sentence_embedding_dimension()
dan1 = models.Dense(in_features=sent_embeddings_dimension, out_features=sent_embeddings_dimension)
dan2 = models.Dense(in_features=sent_embeddings_dimension, out_features=sent_embeddings_dimension)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dan1, dan2])


# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark train dataset")
train_data = SentencesDataset(sts_reader.get_examples('sts-train.csv'), model=model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

logging.info("Read STSbenchmark dev dataset")
dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

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
test_data = SentencesDataset(examples=sts_reader.get_examples("sts-test.csv"), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluator(test_dataloader)

model.evaluate(evaluator)