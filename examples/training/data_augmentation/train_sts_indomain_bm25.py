"""
The script shows how to train Augmented SBERT (In-Domain) strategy for STSb dataset with BM25 sampling.
We utlise easy and practical elasticsearch (https://www.elastic.co/) for BM25 sampling.

Installations:
For this example, elasticsearch to be installed (pip install elasticsearch)
[NOTE] You need to also install ElasticSearch locally on your PC or desktop.
link for download - https://www.elastic.co/downloads/elasticsearch
Or to run it with Docker: https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html

Methodology:
Three steps are followed for AugSBERT data-augmentation with BM25 Sampling - 
    1. Fine-tune cross-encoder (BERT) on gold STSb dataset
    2. Fine-tuned Cross-encoder is used to label on BM25 sampled unlabeled pairs (silver STSb dataset) 
    3. Bi-encoder (SBERT) is finally fine-tuned on both gold + silver STSb dataset

Citation: https://arxiv.org/abs/2010.08240

Usage:
python train_sts_indomain_bm25.py

OR
python train_sts_indomain_bm25.py pretrained_transformer_model_name top_k

python train_sts_indomain_bm25.py bert-base-uncased 3

"""
from torch.utils.data import DataLoader
from sentence_transformers import models, losses, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers import LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from elasticsearch import Elasticsearch
from datetime import datetime
import logging
import csv
import sys
import tqdm
import math
import gzip
import os

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# supressing INFO messages for elastic-search logger
tracer = logging.getLogger('elasticsearch') 
tracer.setLevel(logging.CRITICAL)
es = Elasticsearch()

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'
top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

batch_size = 16
num_epochs = 1
max_seq_length = 128

###### Read Datasets ######

#Check if dataset exsist. If not, download and extract  it
sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

cross_encoder_path = 'output/cross-encoder/stsb_indomain_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
bi_encoder_path = 'output/bi-encoder/stsb_augsbert_BM25_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

###### Cross-encoder (simpletransformers) ######
logging.info("Loading sentence-transformers model: {}".format(model_name))
# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for cross-encoder model
cross_encoder = CrossEncoder(model_name, num_labels=1)


###### Bi-encoder (sentence-transformers) ######
logging.info("Loading bi-encoder model: {}".format(model_name))
# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

bi_encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])


#####################################################################
#
# Step 1: Train cross-encoder model with (gold) STS benchmark dataset
#
#####################################################################

logging.info("Step 1: Train cross-encoder: ({}) with STSbenchmark".format(model_name))

gold_samples = []
dev_samples = []
test_samples = []

with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1

        if row['split'] == 'dev':
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
        elif row['split'] == 'test':
            test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
        else:
            #As we want to get symmetric scores, i.e. CrossEncoder(A,B) = CrossEncoder(B,A), we pass both combinations to the train set
            gold_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
            gold_samples.append(InputExample(texts=[row['sentence2'], row['sentence1']], label=score))


# We wrap gold_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(gold_samples, shuffle=True, batch_size=batch_size)


# We add an evaluator, which evaluates the performance during training
evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='sts-dev')

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the cross-encoder model
cross_encoder.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=cross_encoder_path)

############################################################################
#
# Step 2: Label BM25 sampled STSb (silver dataset) using cross-encoder model
#
############################################################################

#### Top k similar sentences to be retrieved ####
#### Larger the k, bigger the silver dataset ####

index_name = "stsb" # index-name should be in lowercase
logging.info("Step 2.1: Generate STSbenchmark (silver dataset) using top-{} bm25 combinations".format(top_k))

unique_sentences = set()

for sample in gold_samples:
    unique_sentences.update(sample.texts)

unique_sentences = list(unique_sentences) # unique sentences
sent2idx = {sentence: idx for idx, sentence in enumerate(unique_sentences)} # storing id and sentence in dictionary
duplicates = set((sent2idx[data.texts[0]], sent2idx[data.texts[1]]) for data in gold_samples) # not to include gold pairs of sentences again

# Ignore 400 cause by IndexAlreadyExistsException when creating an index
logging.info("Creating elastic-search index - {}".format(index_name))
es.indices.create(index=index_name, ignore=[400]) 

# indexing all sentences
logging.info("Starting to index....")
for sent in unique_sentences:
    response = es.index(
        index = index_name,
        id = sent2idx[sent],
        body = {"sent" : sent})

logging.info("Indexing complete for {} unique sentences".format(len(unique_sentences)))

silver_data = [] 
progress = tqdm.tqdm(unit="docs", total=len(sent2idx))

# retrieval of top-k sentences which forms the silver training data
for sent, idx in sent2idx.items():
    res = es.search(index = index_name, body={"query": {"match": {"sent": sent} } }, size = top_k)
    progress.update(1)
    for hit in res['hits']['hits']:
        if idx != int(hit["_id"]) and (idx, int(hit["_id"])) not in set(duplicates):
            silver_data.append((sent, hit['_source']["sent"]))
            duplicates.add((idx, int(hit["_id"])))

progress.reset()
progress.close()

logging.info("Number of silver pairs generated for STSbenchmark: {}".format(len(silver_data)))
logging.info("Step 2.2: Label STSbenchmark (silver dataset) with cross-encoder: {}".format(model_name))

cross_encoder = CrossEncoder(cross_encoder_path)
silver_scores = cross_encoder.predict(silver_data)

# All model predictions should be between [0,1]
assert all(0.0 <= score <= 1.0 for score in silver_scores)

#################################################################################################
#
# Step 3: Train bi-encoder model with both (gold + silver) STSbenchmark dataset - Augmented SBERT
#
#################################################################################################

logging.info("Step 3: Train bi-encoder: {} with STSbenchmark (gold + silver dataset)".format(model_name))

# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark gold and silver train dataset")
silver_samples = list(InputExample(texts=[data[0], data[1]], label=score) for \
    data, score in zip(silver_data, silver_scores))


train_dataloader = DataLoader(gold_samples + silver_samples, shuffle=True, batch_size=batch_size)
train_loss = losses.CosineSimilarityLoss(model=bi_encoder)

logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

# Configure the training.
warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the bi-encoder model
bi_encoder.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=bi_encoder_path
          )

######################################################################
#
# Evaluate Augmented SBERT performance on STS benchmark (test) dataset
#
######################################################################

# load the stored augmented-sbert model
bi_encoder = SentenceTransformer(bi_encoder_path)
logging.info("Read STSbenchmark test dataset")
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(bi_encoder, output_path=bi_encoder_path)