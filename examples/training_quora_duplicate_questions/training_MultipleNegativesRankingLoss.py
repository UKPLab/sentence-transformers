from torch.utils.data import DataLoader
import math
import sentence_transformers
from sentence_transformers import models, losses, util
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, evaluation
import logging
from datetime import datetime
import sys
import csv
import os
from zipfile import ZipFile
import random

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


#As base model, we use DistilBERT-base that was pre-trained on NLI and STSb data
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

num_epochs = 10

#Increasing the batch size improves the performance for MultipleNegativesRankingLoss. Choose it as large as possible
train_batch_size = 64

dataset_path = 'quora-IR-dataset'
model_save_path = 'output/training_MultipleNegativesRankingLoss-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Check if the dataset exists. If not, download and extract
if not os.path.exists(dataset_path):
    logging.info("Dataset not found. Download")
    zip_save_path = 'quora-IR-dataset.zip'
    util.http_get(url='https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/quora-IR-dataset.zip', path=zip_save_path)
    with ZipFile(zip_save_path, 'r') as zip:
        zip.extractall(dataset_path)


######### Read train data  ##########
train_ids = set()
train_sentences = {}
train_samples = []

# First, we read the question ids (qid) that belong to the train split
with open("quora-IR-dataset/graph/train-questions.tsv", encoding='utf8') as fIn:
    next(fIn)   #Skip header
    for qid in fIn:
        train_ids.add(qid.strip())

# Then, from all sentences, we only extract the train sentences
with open("quora-IR-dataset/graph/sentences.tsv", encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['qid'] in train_ids:
            train_sentences[row['qid']] = row['question']

# Finally, we exract all duplicate relations. These are given in the format (qid1, qid2) indicating that question1 and question2 are duplicates.
# For MultipleNegativesRankingLoss, we only require positive relation, i.e. only sentence pairs that are duplicates. It will
# automatically sample train_batch_size-1 negative pairs.
with open("quora-IR-dataset/graph/duplicates-graph-pairwise.tsv", encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['qid1'] in train_ids and row['qid2'] in train_ids:
            train_samples.append(sentence_transformers.readers.InputExample(texts=[train_sentences[row['qid1']], train_sentences[row['qid2']]], label=1))


# After reading the train_samples, we create a SentencesDataset and a DataLoader
train_dataset = SentencesDataset(train_samples, model=model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model)


################### Development  Evaluators ##################
# We add 3 evaluators, that evaluate the model on Duplicate Questions pair classification,
# Duplicate Questions Mining, and Duplicate Questions Information Retrieval
evaluators = []

###### Classification ######
# Given (quesiton1, question2), is this a duplicate or not?
# The evaluator will compute the embeddings for both questions and then compute
# a cosine similarity. If the similarity is above a threshold, we have a duplicate.
dev_samples = []
with open(os.path.join(dataset_path, "classification/dev_pairs.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        sample = sentence_transformers.readers.InputExample(texts=[row['question1'], row['question2']], label=int(row['is_duplicate']))
        dev_samples.append(sample)


dev_data = SentencesDataset(dev_samples, model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
binary_acc_evaluator = evaluation.BinaryEmbeddingSimilarityEvaluator(dev_dataloader)
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
max_corpus_size = 10000

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
other_questions = {}
with open('dataset/information-retrieval/corpus.tsv', encoding='utf8') as fIn:
    next(fIn) #Skip header
    for line in fIn:
        qid, question = line.strip().split('\t')

        if qid in ir_needed_qids:
            ir_corpus[qid] = question
        else:
            other_questions[qid] = question

# Now, also add some irrelevant questions to fill our corpus
other_qid_list = list(other_questions.keys())
random.shuffle(other_qid_list)

for qid in other_qid_list[0:max(0, max_corpus_size-len(ir_corpus))]:
    ir_corpus[qid] = other_questions[qid]

#Given queries, a corpus and a mapping with relevant documents, the InformationRetrievalEvaluator computes different IR
# metrices. For our use case MRR@k and Accuracy@k are relevant.
ir_eval = evaluation.InformationRetrievalEvaluator(ir_queries, ir_corpus, ir_relevant_docs)


# Create a SequentialEvaluator. This SequentialEvaluator runs all three evaluators in a sequential order.
seq_evaluator = evaluation.SequentialEvaluator(evaluators)


logging.info("Evaluate model without training")
seq_evaluator(model, epoch=0, steps=0, output_path=model_save_path)


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=seq_evaluator,
          epochs=num_epochs,
          warmup_steps=1000,
          output_path=model_save_path,
          output_path_ignore_not_empty=True
          )