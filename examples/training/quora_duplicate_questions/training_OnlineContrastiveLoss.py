"""
This scripts demonstrates how to train a sentence embedding model for question pair classification
with cosine-similarity and a simple threshold.

As dataset, we use Quora Duplicates Questions, where we have labeled pairs of  questions beeing either duplicates (label 1) or non-duplicate (label 0).

As loss function, we use OnlineConstrativeLoss. It reduces the distance between positive pairs, i.e., it pulls the embeddings of positive pairs closer together. For negative pairs, it pushes them further apart.

An issue with constrative loss is, that it might push sentences away that are already well positioned in vector space.
"""

from torch.utils.data import DataLoader
from sentence_transformers import losses, util
from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import csv
import os
from zipfile import ZipFile
import random

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout


#As base model, we use DistilBERT-base that was pre-trained on NLI and STSb data
model = SentenceTransformer('stsb-distilbert-base')
num_epochs = 10
train_batch_size = 64

#As distance metric, we use cosine distance (cosine_distance = 1-cosine_similarity)
distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE

#Negative pairs should have a distance of at least 0.5
margin = 0.5

dataset_path = 'quora-IR-dataset'
model_save_path = 'output/training_OnlineConstrativeLoss-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

os.makedirs(model_save_path, exist_ok=True)

# Check if the dataset exists. If not, download and extract
if not os.path.exists(dataset_path):
    logger.info("Dataset not found. Download")
    zip_save_path = 'quora-IR-dataset.zip'
    util.http_get(url='https://sbert.net/datasets/quora-IR-dataset.zip', path=zip_save_path)
    with ZipFile(zip_save_path, 'r') as zip:
        zip.extractall(dataset_path)


######### Read train data  ##########
# Read train data
train_samples = []
with open(os.path.join(dataset_path, "classification/train_pairs.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        sample = InputExample(texts=[row['question1'], row['question2']], label=int(row['is_duplicate']))
        train_samples.append(sample)



train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.OnlineContrastiveLoss(model=model, distance_metric=distance_metric, margin=margin)



################### Development  Evaluators ##################
# We add 3 evaluators, that evaluate the model on Duplicate Questions pair classification,
# Duplicate Questions Mining, and Duplicate Questions Information Retrieval
evaluators = []

###### Classification ######
# Given (quesiton1, question2), is this a duplicate or not?
# The evaluator will compute the embeddings for both questions and then compute
# a cosine similarity. If the similarity is above a threshold, we have a duplicate.
dev_sentences1 = []
dev_sentences2 = []
dev_labels = []
with open(os.path.join(dataset_path, "classification/dev_pairs.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        dev_sentences1.append(row['question1'])
        dev_sentences2.append(row['question2'])
        dev_labels.append(int(row['is_duplicate']))


binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(dev_sentences1, dev_sentences2, dev_labels)
evaluators.append(binary_acc_evaluator)



###### Duplicate Questions Mining ######
# Given a large corpus of questions, identify all duplicates in that corpus.

# For faster processing, we limit the development corpus to only 10,000 sentences.
max_dev_samples = 10000
dev_sentences = {}
dev_duplicates = []
with open(os.path.join(dataset_path, "duplicate-mining/dev_corpus.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        dev_sentences[row['qid']] = row['question']

        if len(dev_sentences) >= max_dev_samples:
            break

with open(os.path.join(dataset_path, "duplicate-mining/dev_duplicates.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['qid1'] in dev_sentences and row['qid2'] in dev_sentences:
            dev_duplicates.append([row['qid1'], row['qid2']])


# The ParaphraseMiningEvaluator computes the cosine similarity between all sentences and
# extracts a list with the pairs that have the highest similarity. Given the duplicate
# information in dev_duplicates, it then computes and F1 score how well our duplicate mining worked
paraphrase_mining_evaluator = evaluation.ParaphraseMiningEvaluator(dev_sentences, dev_duplicates, name='dev')
evaluators.append(paraphrase_mining_evaluator)


###### Duplicate Questions Information Retrieval ######
# Given a question and a large corpus of thousands questions, find the most relevant (i.e. duplicate) question
# in that corpus.

# For faster processing, we limit the development corpus to only 10,000 sentences.
max_corpus_size = 100000

ir_queries = {}             #Our queries (qid => question)
ir_needed_qids = set()      #QIDs we need in the corpus
ir_corpus = {}              #Our corpus (qid => question)
ir_relevant_docs = {}       #Mapping of relevant documents for a given query (qid => set([relevant_question_ids])

with open(os.path.join(dataset_path, 'information-retrieval/dev-queries.tsv'), encoding='utf8') as fIn:
    next(fIn) #Skip header
    for line in fIn:
        qid, query, duplicate_ids = line.strip().split('\t')
        duplicate_ids = duplicate_ids.split(',')
        ir_queries[qid] = query
        ir_relevant_docs[qid] = set(duplicate_ids)

        for qid in duplicate_ids:
            ir_needed_qids.add(qid)

# First get all needed relevant documents (i.e., we must ensure, that the relevant questions are actually in the corpus
distraction_questions = {}
with open(os.path.join(dataset_path, 'information-retrieval/corpus.tsv'), encoding='utf8') as fIn:
    next(fIn) #Skip header
    for line in fIn:
        qid, question = line.strip().split('\t')

        if qid in ir_needed_qids:
            ir_corpus[qid] = question
        else:
            distraction_questions[qid] = question

# Now, also add some irrelevant questions to fill our corpus
other_qid_list = list(distraction_questions.keys())
random.shuffle(other_qid_list)

for qid in other_qid_list[0:max(0, max_corpus_size-len(ir_corpus))]:
    ir_corpus[qid] = distraction_questions[qid]

#Given queries, a corpus and a mapping with relevant documents, the InformationRetrievalEvaluator computes different IR
# metrices. For our use case MRR@k and Accuracy@k are relevant.
ir_evaluator = evaluation.InformationRetrievalEvaluator(ir_queries, ir_corpus, ir_relevant_docs)

evaluators.append(ir_evaluator)

# Create a SequentialEvaluator. This SequentialEvaluator runs all three evaluators in a sequential order.
# We optimize the model with respect to the score from the last evaluator (scores[-1])
seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])


logger.info("Evaluate model without training")
seq_evaluator(model, epoch=0, steps=0, output_path=model_save_path)


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=seq_evaluator,
          epochs=num_epochs,
          warmup_steps=1000,
          output_path=model_save_path
          )
