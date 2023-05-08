"""
This is an example how to train SentenceTransformers in a multi-task setup.

The system trains BERT on the AllNLI and on the STSbenchmark dataset.
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
import logging
from datetime import datetime
import gzip
import csv
import os

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the dataset
model_name = 'bert-base-uncased'
batch_size = 16
model_save_path = 'output/training_multi-task_'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


#Check if dataset exsist. If not, download and extract  it
nli_dataset_path = 'datasets/AllNLI.tsv.gz'
sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

if not os.path.exists(nli_dataset_path):
    util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)



# Use BERT for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# Convert the dataset to a DataLoader ready for training
logging.info("Read AllNLI train dataset")
label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
train_nli_samples = []
with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'train':
            label_id = label2int[row['label']]
            train_nli_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))


train_dataloader_nli = DataLoader(train_nli_samples, shuffle=True, batch_size=batch_size)
train_loss_nli = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=len(label2int))

logging.info("Read STSbenchmark train dataset")
train_sts_samples = []
dev_sts_samples = []
test_sts_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

        if row['split'] == 'dev':
            dev_sts_samples.append(inp_example)
        elif row['split'] == 'test':
            test_sts_samples.append(inp_example)
        else:
            train_sts_samples.append(inp_example)


train_dataloader_sts = DataLoader(train_sts_samples, shuffle=True, batch_size=batch_size)
train_loss_sts = losses.CosineSimilarityLoss(model=model)


logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_sts_samples, name='sts-dev')

# Configure the training
num_epochs = 4

warmup_steps = math.ceil(len(train_dataloader_sts) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Here we define the two train objectives: train_dataloader_nli with train_loss_nli (i.e., SoftmaxLoss for NLI data)
# and train_dataloader_sts with train_loss_sts (i.e., CosineSimilarityLoss for STSbenchmark data)
# You can pass as many (dataloader, loss) tuples as you like. They are iterated in a round-robin way.
train_objectives = [(train_dataloader_nli, train_loss_nli), (train_dataloader_sts, train_loss_sts)]

# Train the model
model.fit(train_objectives=train_objectives,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )



##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_sts_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)
