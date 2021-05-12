"""
This scripts runs the evaluation (dev & test) for the AskUbuntu dataset

Usage:
python eval_askubuntu.py [sbert_model_name_or_path]
"""

from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import util, evaluation
import logging
import os
import gzip
import sys

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

model = SentenceTransformer(sys.argv[1])


################# Download AskUbuntu and extract training corpus  #################
askubuntu_folder = 'askubuntu'
training_corpus = os.path.join(askubuntu_folder, 'train.unsupervised.txt')


## Download the AskUbuntu dataset from https://github.com/taolei87/askubuntu
for filename in ['text_tokenized.txt.gz', 'dev.txt', 'test.txt', 'train_random.txt']:
    filepath = os.path.join(askubuntu_folder, filename)
    if not os.path.exists(filepath):
        util.http_get('https://github.com/taolei87/askubuntu/raw/master/'+filename, filepath)

# Read the corpus
corpus = {}
dev_test_ids = set()
with gzip.open(os.path.join(askubuntu_folder, 'text_tokenized.txt.gz'), 'rt', encoding='utf8') as fIn:
    for line in fIn:
        splits = line.strip().split("\t")
        id = splits[0]
        title = splits[1]
        corpus[id] = title

# Read dev & test dataset
def read_eval_dataset(filepath):
    dataset = []
    with open(filepath) as fIn:
        for line in fIn:
            query_id, relevant_id, candidate_ids, bm25_scores = line.strip().split("\t")
            if len(relevant_id) == 0:   #Skip examples without relevant entries
                continue

            relevant_id = relevant_id.split(" ")
            candidate_ids = candidate_ids.split(" ")
            negative_ids = set(candidate_ids) - set(relevant_id)
            dataset.append({
                'query': corpus[query_id],
                'positive': [corpus[pid] for pid in relevant_id],
                'negative': [corpus[pid] for pid in negative_ids]
            })
            dev_test_ids.add(query_id)
            dev_test_ids.update(candidate_ids)
    return dataset

dev_dataset = read_eval_dataset(os.path.join(askubuntu_folder, 'dev.txt'))
test_dataset = read_eval_dataset(os.path.join(askubuntu_folder, 'test.txt'))




# Create a dev evaluator
dev_evaluator = evaluation.RerankingEvaluator(dev_dataset, name="AskUbuntu dev")

logging.info("Dev performance before training")
dev_evaluator(model)

test_evaluator = evaluation.RerankingEvaluator(test_dataset, name="AskUbuntu test")
logging.info("Test performance before training")
test_evaluator(model)
