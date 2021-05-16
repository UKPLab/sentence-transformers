"""
This examples show how to train a Cross-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

In this example we use a knowledge distillation setup. Sebastian Hofstätter et al. trained in https://arxiv.org/abs/2010.02666
an ensemble of large Transformer models for the MS MARCO datasets and combines the scores from a BERT-base, BERT-large, and ALBERT-large model.

We use the logits scores from the ensemble to train a smaller model. We found that the MiniLM model gives the best performance while
offering the highest speed.

The resulting Cross-Encoder can then be used for passage re-ranking: You retrieve for example 100 passages
for a given query, for example with ElasticSearch, and pass the query+retrieved_passage to the CrossEncoder
for scoring. You sort the results then according to the output of the CrossEncoder.

This gives a significant boost compared to out-of-the-box ElasticSearch / BM25 ranking.

Running this script:
python train_cross-encoder-v2.py
"""
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
import torch

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


#First, we define the transformer model we want to fine-tune
model_name = 'microsoft/MiniLM-L12-H384-uncased'
train_batch_size = 32
num_epochs = 1
model_save_path = 'output/training_ms-marco_cross-encoder-v2-'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



#We set num_labels=1 and set the activation function to Identiy, so that we get the raw logits
model = CrossEncoder(model_name, num_labels=1, max_length=512, default_activation_function=torch.nn.Identity())


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


### Now we create our  dev data
train_samples = []
dev_samples = {}

# We use 200 random queries from the train set for evaluation during training
# Each query has at least one relevant and up to 200 irrelevant (negative) passages
num_dev_queries = 200
num_max_dev_negatives = 200

# msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz and msmarco-qidpidtriples.rnd-shuf.train.tsv.gz is a randomly
# shuffled version of qidpidtriples.train.full.2.tsv.gz from the MS Marco website
# We extracted in the train-eval split 500 random queries that can be used for evaluation during training
train_eval_filepath = os.path.join(data_folder, 'msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz')
if not os.path.exists(train_eval_filepath):
    logging.info("Download "+os.path.basename(train_eval_filepath))
    util.http_get('https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz', train_eval_filepath)

with gzip.open(train_eval_filepath, 'rt') as fIn:
    for line in fIn:
        qid, pos_id, neg_id = line.strip().split()

        if qid not in dev_samples and len(dev_samples) < num_dev_queries:
            dev_samples[qid] = {'query': queries[qid], 'positive': set(), 'negative': set()}

        if qid in dev_samples:
            dev_samples[qid]['positive'].add(corpus[pos_id])

            if len(dev_samples[qid]['negative']) < num_max_dev_negatives:
                dev_samples[qid]['negative'].add(corpus[neg_id])

dev_qids = set(dev_samples.keys())

# Read our training file
# As input examples, we provide the (query, passage) pair together with the logits score from the teacher ensemble
teacher_logits_filepath = os.path.join(data_folder, 'bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv')
train_samples = []
if not os.path.exists(teacher_logits_filepath):
    util.http_get('https://zenodo.org/record/4068216/files/bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv?download=1', teacher_logits_filepath)

with open(teacher_logits_filepath) as fIn:
    for line in fIn:
        pos_score, neg_score, qid, pid1, pid2 = line.strip().split("\t")

        if qid in dev_qids: #Skip queries in our dev dataset
            continue

        train_samples.append(InputExample(texts=[queries[qid], corpus[pid1]], label=float(pos_score)))
        train_samples.append(InputExample(texts=[queries[qid], corpus[pid2]], label=float(neg_score)))

# We create a DataLoader to load our train samples
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)

# We add an evaluator, which evaluates the performance during training
# It performs a classification task and measures scores like F1 (finding relevant passages) and Average Precision
evaluator = CERerankingEvaluator(dev_samples, name='train-eval')

# Configure the training
warmup_steps = 5000
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_dataloader=train_dataloader,
          loss_fct=torch.nn.MSELoss(),
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=5000,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          optimizer_params={'lr': 7e-6},
          use_amp=True)

#Save latest model
model.save(model_save_path+'-latest')