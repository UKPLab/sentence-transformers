"""
This examples show how to train a Bi-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)

Negative passage are hard negative examples, that were mined using different dense embedding methods and lexical search methods.
Each positive and negative passage comes with a score from a Cross-Encoder. This allows denoising, i.e. removing false negative
passages that are actually relevant for the query.

With a distilbert-base-uncased model, it should achieve a performance of about 33.79 MRR@10 on the MSMARCO Passages Dev-Corpus

Running this script:
python train_bi-encoder-v3.py
"""
import argparse
import gzip
import json
import logging
import os
import pickle
import random
import tarfile
from datetime import datetime

import torch.cuda
import tqdm
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, losses, InputExample, evaluation
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from accelerate import Accelerator

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--model_name", required=True)
parser.add_argument("--steps_per_epoch", default=None, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument("--negs_to_use", default=None,
                    help="From which systems should negatives be used? Multiple systems seperated by comma. None = all")
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negs_per_system", default=5, type=int)
parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
parser.add_argument("--use_all_queries", default=False, action="store_true")
parser.add_argument("--ce_score_margin", default=3.0, type=float)
args = parser.parse_args()

print(args)

accelerator = Accelerator()

# The  model we want to fine-tune
model_name = args.model_name

train_batch_size = args.train_batch_size  # Increasing the train batch size improves the model performance, but requires more GPU memory
max_seq_length = args.max_seq_length  # Max length for passages. Increasing it, requires more GPU memory
ce_score_margin = args.ce_score_margin  # Margin for the CrossEncoder score between negative and positive passages
num_negs_per_system = args.num_negs_per_system  # We used different systems to mine hard negatives. Number of hard negatives to add from each system
num_epochs = args.epochs  # Number of epochs we want to train

# Load our embedding model
if args.use_pre_trained_model:
    logging.info("use pretrained SBERT model")
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
else:
    logging.info("Create new SBERT model")
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model_save_path = 'output/train_bi-encoder-mnrl-{}-margin_{:.1f}-{}'.format(model_name.replace("/", "-"),
                                                                            ce_score_margin, datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"))

### Now we read the MS Marco dataset
data_folder = 'msmarco-data'

collection_filepath = os.path.join(data_folder, 'collection.tsv')
queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
ce_scores_file = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')
hard_negatives_filepath = os.path.join(data_folder, 'msmarco-hard-negatives.jsonl.gz')

if accelerator.is_main_process:
    # Downloads
    #### Read the corpus files, that contain all the passages. Store them in the corpus dict
    corpus = {}  # dict in the format: passage_id -> passage. Stores all existent passages
    if not os.path.exists(collection_filepath):
        tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
        if not os.path.exists(tar_filepath):
            logging.info("Download collection.tar.gz")
            util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)

    if not os.path.exists(queries_filepath):
        tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
        if not os.path.exists(tar_filepath):
            logging.info("Download queries.tar.gz")
            util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)

    # Load a dict (qid, pid) -> ce_score that maps query-ids (qid) and paragraph-ids (pid)
    # to the CrossEncoder score computed by the cross-encoder/ms-marco-MiniLM-L-6-v2 model
    if not os.path.exists(ce_scores_file):
        logging.info("Download cross-encoder scores file")
        util.http_get(
            'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz',
            ce_scores_file)

    # As training data we use hard-negatives that have been mined using various systems
    if not os.path.exists(hard_negatives_filepath):
        logging.info("Download cross-encoder scores file")
        util.http_get(
            'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz',
            hard_negatives_filepath)

### Read the train queries, store in queries dict
queries = {}  # dict in the format: query_id -> query. Stores all training queries

with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        qid = int(qid)
        queries[qid] = query

logging.info("Read corpus: collection.tsv")
with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        pid = int(pid)
        corpus[pid] = passage

logging.info("Load CrossEncoder scores dict")
with gzip.open(ce_scores_file, 'rb') as fIn:
    ce_scores = pickle.load(fIn)

logging.info("Read hard negatives train file")
train_queries = {}
negs_to_use = None
with gzip.open(hard_negatives_filepath, 'rt') as fIn:
    for line in tqdm.tqdm(fIn):
        data = json.loads(line)

        # Get the positive passage ids
        qid = data['qid']
        pos_pids = data['pos']

        if len(pos_pids) == 0:  # Skip entries without positives passages
            continue

        pos_min_ce_score = min([ce_scores[qid][pid] for pid in data['pos']])
        ce_score_threshold = pos_min_ce_score - ce_score_margin

        # Get the hard negatives
        neg_pids = set()
        if negs_to_use is None:
            if args.negs_to_use is not None:  # Use specific system for negatives
                negs_to_use = args.negs_to_use.split(",")
            else:  # Use all systems
                negs_to_use = list(data['neg'].keys())
            logging.info("Using negatives from the following systems: {}".format(", ".join(negs_to_use)))

        for system_name in negs_to_use:
            if system_name not in data['neg']:
                continue

            system_negs = data['neg'][system_name]
            negs_added = 0
            for pid in system_negs:
                if ce_scores[qid][pid] > ce_score_threshold:
                    continue

                if pid not in neg_pids:
                    neg_pids.add(pid)
                    negs_added += 1
                    if negs_added >= num_negs_per_system:
                        break

        if args.use_all_queries or (len(pos_pids) > 0 and len(neg_pids) > 0):
            train_queries[data['qid']] = {'qid': data['qid'], 'query': queries[data['qid']], 'pos': pos_pids,
                                          'neg': neg_pids}

logging.info("Train queries: {}".format(len(train_queries)))


# We create a custom MSMARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.
class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']

        pos_id = query['pos'].pop(0)  # Pop positive and add at end
        pos_text = self.corpus[pos_id]
        query['pos'].append(pos_id)

        neg_id = query['neg'].pop(0)  # Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query['neg'].append(neg_id)

        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.queries)

# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(train_queries, corpus=corpus)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=args.warmup_steps,
          use_amp=True,
          checkpoint_path=model_save_path,
          checkpoint_save_steps=len(train_dataloader),
          optimizer_params={'lr': args.lr},
          show_progress_bar=True,
          steps_per_epoch=args.steps_per_epoch,
          accelerator=accelerator
          )

# Save the model
model.save(model_save_path)

# Evaluate
### Load eval data
dev_corpus_max_size = 0
collection_filepath = os.path.join(data_folder, 'collection.tsv')
dev_queries_file = os.path.join(data_folder, 'queries.dev.small.tsv')
qrels_filepath = os.path.join(data_folder, 'qrels.dev.tsv')

dev_corpus = {}  # Our corpus pid => passage
dev_queries = {}  # Our dev queries. qid => query
dev_rel_docs = {}  # Mapping qid => set with relevant pids
needed_pids = set()  # Passage IDs we need
needed_qids = set()  # Query IDs we need

### Download files if needed
if accelerator.is_main_process:
    if not os.path.exists(collection_filepath) or not os.path.exists(dev_queries_file):
        tar_filepath = os.path.join(data_folder, 'collectionandqueries.tar.gz')
        if not os.path.exists(tar_filepath):
            logging.info("Download: " + tar_filepath)
            util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz', tar_filepath)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)

    if not os.path.exists(qrels_filepath):
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv', qrels_filepath)

# Load the 6980 dev queries
with open(dev_queries_file, encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        dev_queries[qid] = query.strip()

# Load which passages are relevant for which queries
with open(qrels_filepath) as fIn:
    for line in fIn:
        qid, _, pid, _ = line.strip().split('\t')

        if qid not in dev_queries:
            continue

        if qid not in dev_rel_docs:
            dev_rel_docs[qid] = set()
        dev_rel_docs[qid].add(pid)

        needed_pids.add(pid)
        needed_qids.add(qid)

# Read passages
with open(collection_filepath, encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        passage = passage

        if pid in needed_pids or dev_corpus_max_size <= 0 or len(dev_corpus) <= dev_corpus_max_size:
            dev_corpus[pid] = passage.strip()

#only performing evaluation from one process
if accelerator.is_main_process:
    ir_evaluator = evaluation.InformationRetrievalEvaluator(dev_queries, dev_corpus, dev_rel_docs,
                                                            show_progress_bar=True,
                                                            corpus_chunk_size=100000,
                                                            precision_recall_at_k=[10],
                                                            name="msmarco dev")
    ir_evaluator(model, num_proc=torch.cuda.device_count())
