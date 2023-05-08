"""
This examples trains a CrossEncoder for the Quora Duplicate Questions Detection task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it output a continious labels 0...1 to indicate the similarity between the input pair.

It does NOT produce a sentence embedding and does NOT work for individual sentences.

Usage:
python training_quora_duplicate_questions.py

"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import os
import gzip
import csv
from zipfile import ZipFile

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout



#Check if dataset exsist. If not, download and extract  it
dataset_path = 'quora-dataset/'

if not os.path.exists(dataset_path):
    logger.info("Dataset not found. Download")
    zip_save_path = 'quora-IR-dataset.zip'
    util.http_get(url='https://sbert.net/datasets/quora-IR-dataset.zip', path=zip_save_path)
    with ZipFile(zip_save_path, 'r') as zip:
        zip.extractall(dataset_path)


# Read the quora dataset split for classification
logger.info("Read train dataset")
train_samples = []
with open(os.path.join(dataset_path, 'classification', 'train_pairs.tsv'), 'r', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        train_samples.append(InputExample(texts=[row['question1'], row['question2']], label=int(row['is_duplicate'])))
        train_samples.append(InputExample(texts=[row['question2'], row['question1']], label=int(row['is_duplicate'])))


logger.info("Read dev dataset")
dev_samples = []
with open(os.path.join(dataset_path, 'classification', 'dev_pairs.tsv'), 'r', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        dev_samples.append(InputExample(texts=[row['question1'], row['question2']], label=int(row['is_duplicate'])))


#Configuration
train_batch_size = 16
num_epochs = 4
model_save_path = 'output/training_quora-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


#We use distilroberta-base with a single label, i.e., it will output a value between 0 and 1 indicating the similarity of the two questions
model = CrossEncoder('distilroberta-base', num_labels=1)

# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)


# We add an evaluator, which evaluates the performance during training
evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples, name='Quora-dev')


# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logger.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=5000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


