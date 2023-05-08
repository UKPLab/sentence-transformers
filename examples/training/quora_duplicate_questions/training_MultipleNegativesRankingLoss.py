"""
This scripts demonstrates how to train a sentence embedding model for Information Retrieval.

As dataset, we use Quora Duplicates Questions, where we have pairs of duplicate questions.

As loss function, we use MultipleNegativesRankingLoss. Here, we only need positive pairs, i.e., pairs of sentences/texts that are considered to be relevant. Our dataset looks like this (a_1, b_1), (a_2, b_2), ... with a_i / b_i a text and (a_i, b_i) are relevant (e.g. are duplicates).

MultipleNegativesRankingLoss takes a random subset of these, for example (a_1, b_1), ..., (a_n, b_n). a_i and b_i are considered to be relevant and should be close in vector space. All other b_j (for i != j) are negative examples and the distance between a_i and b_j should be maximized. Note: MultipleNegativesRankingLoss only works if a random b_j is likely not to be relevant for a_i. This is the case for our duplicate questions dataset: If a sample randomly b_j, it is unlikely to be a duplicate of a_i.


The model we get works well for duplicate questions mining and for duplicate questions information retrieval. For question pair classification, other losses (like OnlineConstrativeLoss) work better.
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

#Training for multiple epochs can be beneficial, as in each epoch a mini-batch is sampled differently
#hence, we get different negatives for each positive
num_epochs = 10

#Increasing the batch size improves the performance for MultipleNegativesRankingLoss. Choose it as large as possible
#I achieved the good results with a batch size of 300-350 (requires about 30 GB of GPU memory)
train_batch_size = 64

dataset_path = 'quora-IR-dataset'
model_save_path = 'output/training_MultipleNegativesRankingLoss-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

os.makedirs(model_save_path, exist_ok=True)

# Check if the dataset exists. If not, download and extract
if not os.path.exists(dataset_path):
    logger.info("Dataset not found. Download")
    zip_save_path = 'quora-IR-dataset.zip'
    util.http_get(url='https://sbert.net/datasets/quora-IR-dataset.zip', path=zip_save_path)
    with ZipFile(zip_save_path, 'r') as zip:
        zip.extractall(dataset_path)


######### Read train data  ##########
train_samples = []
with open(os.path.join(dataset_path, "classification/train_pairs.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['is_duplicate'] == '1':
            train_samples.append(InputExample(texts=[row['question1'], row['question2']], label=1))
            train_samples.append(InputExample(texts=[row['question2'], row['question1']], label=1)) #if A is a duplicate of B, then B is a duplicate of A


# After reading the train_samples, we create a DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model)


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
