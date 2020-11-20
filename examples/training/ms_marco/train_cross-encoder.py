"""
This examples show how to train a Cross-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

The query and the passage are passed simoultanously to a Transformer network. The network then returns
a score between 0 and 1 how relevant the passage is for a given query.

The resulting Cross-Encoder can then be used for passage re-ranking: You retrieve for example 100 passages
for a given query, for example with ElasticSearch, and pass the query+retrieved_passage to the CrossEncoder
for scoring. You sort the results then according to the output of the CrossEncoder.

This gives a significant boost compared to out-of-the-box ElasticSearch / BM25 ranking.

Running this script:
python train_cross-encoder.py
"""
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers import InputExample
import logging
from datetime import datetime
import gzip
import sys
import os
import tarfile

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


#First, we define the transformer model we want to fine-tune
model_name = 'google/electra-small-discriminator'
train_batch_size = 32
num_epochs = 1
model_save_path = 'output/training_ms-marco_cross-encoder-'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

max_train_size = 2e7    #We limit our training samples to 20 Million. The dataset provides about 400 Million triplets (query, positive passage, negative passage)

# We train the network with as a binary label task
# Given [query, passage] is the label 0 = irrelevant or 1 = relevant?
# We use a positive-to-negative ratio: For 1 positive sample (label 1) we include 4 negative samples (label 0)
# in our training setup. For the negative samples, we use the triplets provided by MS Marco that
# specify (query, positive sample, negative sample).
pos_neg_ration = 4

#We set num_labels=1, which predicts a continous score between 0 and 1
model = CrossEncoder(model_name, num_labels=1, max_length=512)


### Now we read the MS Marco dataset
data_folder = 'msmarco-data'
os.makedirs(data_folder, exist_ok=True)




#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}
collection_filepath = os.path.join(data_folder, 'collection.tsv')
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download collection.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        corpus[pid] = passage


### Read the train queries, store in queries dict
queries = {}
queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
if not os.path.exists(queries_filepath):
    tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download queries.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)


with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query



### Now we create our training & dev data
### As dev data, we use the first 10k samples from the training corpus
num_dev_samples = 10000

cnt = 0
train_samples = []
dev_samples = []
dev_qids = set()    #Stores which query ids (qids) have been included in dev_samples. So that we don't use them in the training set

triplets_filepath = os.path.join(data_folder, 'qidpidtriples.train.full.2.tsv.gz')
if not os.path.exists(triplets_filepath):
    util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz', triplets_filepath)

with gzip.open(triplets_filepath, 'rt') as fIn:
    for line in fIn:
        cnt += 1
        qid, pos_id, neg_id = line.strip().split()
        query = queries[qid]

        #Select alternatively positive and negative samples according to our positive-to-negative ratio.
        if (cnt % (pos_neg_ration+1)) == 0:
            passage = corpus[pos_id]
            label = 1
        else:
            passage = corpus[neg_id]
            label = 0

        example = InputExample(texts=[query, passage], label=label)
        if len(dev_samples) <= num_dev_samples:
            dev_samples.append(example)
            dev_qids.add(qid)
        elif qid not in dev_qids:
            train_samples.append(example)

        if len(train_samples) >= max_train_size:
            break

# We create a DataLoader to load our train samples
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# We add an evaluator, which evaluates the performance during training
# It performs a classification task and measures scores like F1 (finding relevant passages) and Average Precision
evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples, name='MS-Marco')

# Configure the training
warmup_steps = 5000
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=5000,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=True)

#Save latest model
model.save(model_save_path+'-latest')