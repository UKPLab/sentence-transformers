"""
This examples show how to train a Bi-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)

Negative passage are hard negative examples, that where retrieved by lexical search. We use the negative
passages (the triplets) that are provided by the MS MARCO dataset.

Running this script:
python train_bi-encoder.py
"""
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
from collections import defaultdict
from torch.utils.data import IterableDataset


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# The  model we want to fine-tune
model_name = 'distilroberta-base'

train_batch_size = 64           #Increasing the train batch size improves the model performance, but requires more GPU memory

num_dev_queries = 500           #Number of queries we want to use to evaluate the performance while training
num_max_dev_negatives = 200     #For every dev query, we use up to 200 hard negatives and add them to the dev corpus

# We construct the SentenceTransformer bi-encoder from scratch
word_embedding_model = models.Transformer(model_name, max_seq_length=350)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model_save_path = 'output/training_ms-marco_bi-encoder-'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


### Now we read the MS Marco dataset
data_folder = 'msmarco-data'
os.makedirs(data_folder, exist_ok=True)

#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}         #dict in the format: passage_id -> passage. Stores all existent passages
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
queries = {}        #dict in the format: query_id -> query. Stores all training queries
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


# msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz and msmarco-qidpidtriples.rnd-shuf.train.tsv.gz is a randomly
# shuffled version of qidpidtriples.train.full.2.tsv.gz from the MS Marco website
# We extracted in the train-eval split 500 random queries that can be used for evaluation during training
train_eval_filepath = os.path.join(data_folder, 'msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz')
if not os.path.exists(train_eval_filepath):
    logging.info("Download "+os.path.basename(train_eval_filepath))
    util.http_get('https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz', train_eval_filepath)

dev_queries = {}
dev_corpus = {}
dev_rel_docs = {}

num_negatives = defaultdict(int)

with gzip.open(train_eval_filepath, 'rt') as fIn:
    for line in fIn:
        qid, pos_id, neg_id = line.strip().split()

        if len(dev_queries) <= num_dev_queries or qid in dev_queries:
            dev_queries[qid] = queries[qid]

            #Ensure the corpus has the positive
            dev_corpus[pos_id] = corpus[pos_id]

            if qid not in dev_rel_docs:
                dev_rel_docs[qid] = set()

            dev_rel_docs[qid].add(pos_id)

            if num_negatives[qid] < num_max_dev_negatives:
                dev_corpus[neg_id] = corpus[neg_id]
                num_negatives[qid] += 1

logging.info("Dev queries: {}".format(len(dev_queries)))
logging.info("Dev Corpus: {}".format(len(dev_corpus)))


# Create the evaluator that is called during training
ir_evaluator = evaluation.InformationRetrievalEvaluator(dev_queries, dev_corpus, dev_rel_docs, name='ms-marco-train_eval')

# Read our training file. qidpidtriples consists of triplets (qid, positive_pid, negative_pid)
train_filepath = os.path.join(data_folder, 'msmarco-qidpidtriples.rnd-shuf.train.tsv.gz')
if not os.path.exists(train_filepath):
    logging.info("Download "+os.path.basename(train_filepath))
    util.http_get('https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train.tsv.gz', train_filepath)


#We load the qidpidtriples file on-the-fly by using a custom IterableDataset class
class TripletsDataset(IterableDataset):
    def __init__(self, model, queries, corpus, triplets_file):
        self.model = model
        self.queries = queries
        self.corpus = corpus
        self.triplets_file = triplets_file

    def __iter__(self):
        with gzip.open(self.triplets_file, 'rt') as fIn:
            for line in fIn:
                qid, pos_id, neg_id = line.strip().split()
                query_text = self.queries[qid]
                pos_text = self.corpus[pos_id]
                neg_text = self.corpus[neg_id]
                yield InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return 397226027

# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = TripletsDataset(model=model, queries=queries, corpus=corpus, triplets_file=train_filepath)
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=ir_evaluator,
          epochs=1,
          warmup_steps=1000,
          output_path=model_save_path,
          evaluation_steps=5000,
          use_amp=True
          )