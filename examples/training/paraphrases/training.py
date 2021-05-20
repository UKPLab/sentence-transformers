import math
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
from MultiDatasetDataLoader import MultiDatasetDataLoader

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

model_name = 'distilroberta-base'
num_epochs = 1
sts_dataset_path = 'data-eval/stsbenchmark.tsv.gz'
batch_size_pairs = 384
batch_size_triplets = 256
max_seq_length = 128
use_amp = True                  #Set to False, if you use a CPU or your GPU does not support FP16 operations
evaluation_steps = 500
warmup_steps = 500

#####

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)


# Save path of the model
model_save_path = 'output/training_paraphrases_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



## SentenceTransformer model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

datasets = []
for filepath in sys.argv[1:]:
    dataset = []
    with_guid = 'with-guid' in filepath     #Some datasets have a guid in the first column

    with gzip.open(filepath, 'rt', encoding='utf8') as fIn:
        for line in fIn:
            splits = line.strip().split("\t")
            if with_guid:
                guid = splits[0]
                texts = splits[1:]
            else:
                guid = None
                texts = splits

            dataset.append(InputExample(texts=texts, guid=guid))

    datasets.append(dataset)


train_dataloader = MultiDatasetDataLoader(datasets, batch_size_pairs=batch_size_pairs, batch_size_triplets=batch_size_triplets)



# Our training loss
train_loss = losses.MultipleNegativesRankingLoss(model)



#Read STSbenchmark dataset and use it as development set
logging.info("Read STSbenchmark dev dataset")
dev_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'dev':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

# Configure the training
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=evaluation_steps,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=use_amp,
          checkpoint_path=model_save_path,
          checkpoint_save_steps=1000,
          checkpoint_save_total_limit=3
          )
