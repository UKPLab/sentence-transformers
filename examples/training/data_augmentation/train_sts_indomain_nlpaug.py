"""
The script shows how to train Augmented SBERT (In-Domain) strategy for STSb dataset with nlp textual augmentation.
We utilise nlpaug (https://github.com/makcedward/nlpaug) for data augmentation strategies over a single sentence.

We chose synonym replacement for our example with (can be extended to other techniques) -
    1. Word-embeddings (word2vec) 
    2. WordNet
    3. Contextual word-embeddings (BERT)

Methodology:
Take a gold STSb pair, like (A, B, 0.6) Then replace synonyms in A and B, which gives you (A', B', 0.6) 
These are the silver data and SBERT is finally trained on (gold + silver) STSb data.

Additional requirements:
pip install nlpaug

Information:
We went over the nlpaug package and found from our experience, the commonly used and effective technique
is synonym replacement with words. However feel free to use any textual data augmentation mentioned
in the example - (https://github.com/makcedward/nlpaug/blob/master/example/textual_augmenter.ipynb)

You could also extend the easy data augmentation methods for other languages too, a good example can be
found here - (https://github.com/makcedward/nlpaug/blob/master/example/textual_language_augmenter.ipynb)


Citation: https://arxiv.org/abs/2010.08240

Usage:
python train_sts_indomain_nlpaug.py
"""
from torch.utils.data import DataLoader
import torch
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader, InputExample
import nlpaug.augmenter.word as naw
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import tqdm

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
num_epochs = 1

###### Read Datasets ######

#Check if dataset exsist. If not, download and extract  it
sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)


model_save_path = 'output/bi-encoder/stsb_indomain_eda_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

###### Bi-encoder (sentence-transformers) ######
logging.info("Loading SBERT model: {}".format(model_name))
# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Convert the dataset to a DataLoader ready for training
gold_samples = []
dev_samples = []
test_samples = []

with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

        if row['split'] == 'dev':
            dev_samples.append(inp_example)
        elif row['split'] == 'test':
            test_samples.append(inp_example)
        else:
            gold_samples.append(inp_example)

##################################################################################
#
# Data Augmentation: Synonym Replacement with word2vec, BERT, WordNet using nlpaug
#
##################################################################################

logging.info("Starting with synonym replacement...")

#### Synonym replacement using Word2Vec ####
# Download the word2vec pre-trained Google News corpus (GoogleNews-vectors-negative300.bin)
# link: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

# aug = naw.WordEmbsAug(
#     model_type='word2vec', model_path=model_dir+'GoogleNews-vectors-negative300.bin',
#     action="substitute")

#### Synonym replacement using WordNet ####
# aug = naw.SynonymAug(aug_src='wordnet')

#### Synonym replacement using BERT ####
aug = naw.ContextualWordEmbsAug(
    model_path=model_name, action="insert", device=device)

silver_samples = []
progress = tqdm.tqdm(unit="docs", total=len(gold_samples))

for sample in gold_samples:
    augmented_texts = aug.augment(sample.texts)
    inp_example = InputExample(texts=augmented_texts, label=sample.label)
    silver_samples.append(inp_example)
    progress.update(1)

progress.reset()
progress.close()
logging.info("Textual augmentation completed....")
logging.info("Number of silver pairs generated: {}".format(len(silver_samples)))

###################################################################
#
# Train SBERT model with both (gold + silver) STS benchmark dataset
#
###################################################################

logging.info("Read STSbenchmark (gold + silver) training dataset")
train_dataloader = DataLoader(gold_samples + silver_samples, shuffle=True, batch_size=batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)


logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')


# Configure the training.
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the SBERT model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )

##########################################################
#
# Evaluate SBERT performance on STS benchmark test dataset
#
##########################################################

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)