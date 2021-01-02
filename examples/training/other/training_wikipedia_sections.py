"""
This script trains sentence transformers with a triplet loss function.

As corpus, we use the wikipedia sections dataset that was describd by Dor et al., 2018, Learning Thematic Similarity Metric Using Triplet Networks.
"""

from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler, losses, models, util
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import TripletEvaluator
from datetime import datetime
from zipfile import ZipFile

import csv
import logging
import os


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = 'distilbert-base-uncased'

dataset_path = 'datasets/wikipedia-sections'
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path, exist_ok=True)
    filepath = os.path.join(dataset_path, 'wikipedia-sections-triplets.zip')
    util.http_get('https://sbert.net/datasets/wikipedia-sections-triplets.zip', filepath)
    with ZipFile(filepath, 'r') as zip:
        zip.extractall(dataset_path)


### Create a torch.DataLoader that passes training batch instances to our model
train_batch_size = 16
output_path = "output/training-wikipedia-sections-"+model_name+"-"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
num_epochs = 1


### Configure sentence transformers for training and train on the provided dataset
# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


logger.info("Read Triplet train dataset")
train_examples = []
with open(os.path.join(dataset_path, 'train.csv'), encoding="utf-8") as fIn:
    reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        train_examples.append(InputExample(texts=[row['Sentence1'], row['Sentence2'], row['Sentence3']], label=0))



train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.TripletLoss(model=model)

logger.info("Read Wikipedia Triplet dev dataset")
dev_examples = []
with open(os.path.join(dataset_path, 'validation.csv'), encoding="utf-8") as fIn:
    reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        dev_examples.append(InputExample(texts=[row['Sentence1'], row['Sentence2'], row['Sentence3']]))

        if len(dev_examples) >= 1000:
            break

evaluator = TripletEvaluator.from_input_examples(dev_examples, name='dev')


warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) #10% of train data


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=output_path)

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

logger.info("Read test examples")
test_examples = []
with open(os.path.join(dataset_path, 'test.csv'), encoding="utf-8") as fIn:
    reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        test_examples.append(InputExample(texts=[row['Sentence1'], row['Sentence2'], row['Sentence3']]))


model = SentenceTransformer(output_path)
test_evaluator = TripletEvaluator.from_input_examples(test_examples, name='test')
test_evaluator(model, output_path=output_path)

