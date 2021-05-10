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
import random
import glob

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

model_name = 'distilroberta-base'
num_epochs = 1
sts_dataset_path = 'data-eval/stsbenchmark.tsv.gz'
batch_size_pairs = 256
batch_size_triplets = 256
max_seq_length = 100
use_amp = True
evaluation_steps = 500
warmup_steps = 500

#####

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)


# Save path of the model
model_save_path = 'output/training_paraphrases_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class MultiDatasetDataLoader:
    def __init__(self, datasets, batch_size_pairs, batch_size_triplets = None, dataset_size_temp=5):
        self.allow_swap = True
        self.batch_size_pairs = batch_size_pairs
        self.batch_size_triplets = batch_size_pairs if batch_size_triplets is None else batch_size_triplets

        # Compute dataset weights
        self.dataset_lengths = list(map(len, datasets))
        self.dataset_lengths_sum = sum(self.dataset_lengths)

        weights = []
        for dataset in datasets:
            prob = len(dataset) / self.dataset_lengths_sum
            weights.append(max(1, int(math.pow(prob, 1 / dataset_size_temp) * 1000)))

        logging.info("Dataset lenghts and weights: {}".format(list(zip(self.dataset_lengths, weights))))

        self.dataset_idx = []
        self.dataset_idx_pointer = 0

        for idx, weight in enumerate(weights):
            self.dataset_idx.extend([idx] * weight)
        random.shuffle(self.dataset_idx)

        self.datasets = []
        for dataset in datasets:
            random.shuffle(dataset)
            self.datasets.append({
                'elements': dataset,
                'pointer': 0,
            })

    def __iter__(self):
        for _ in range(int(self.__len__())):
            #Select dataset
            if self.dataset_idx_pointer >= len(self.dataset_idx):
                self.dataset_idx_pointer = 0
                random.shuffle(self.dataset_idx)

            dataset_idx = self.dataset_idx[self.dataset_idx_pointer]
            self.dataset_idx_pointer += 1

            #Select batch from this dataset
            dataset = self.datasets[dataset_idx]
            batch_size = self.batch_size_pairs if len(dataset['elements'][0].texts) == 2 else self.batch_size_triplets

            batch = []
            texts_in_batch = set()
            while len(batch) < batch_size:
                example = dataset['elements'][dataset['pointer']]

                valid_example = True
                for text in example.texts:
                    if text.strip().lower() in texts_in_batch:
                        valid_example = False
                        break

                if valid_example:
                    if self.allow_swap and random.random() > 0.5:
                        example.texts[0], example.texts[1] = example.texts[1], example.texts[0]

                    batch.append(example)
                    for text in example.texts:
                        texts_in_batch.add(text.strip().lower())

                dataset['pointer'] += 1
                if dataset['pointer'] >= len(dataset['elements']):
                    dataset['pointer'] = 0
                    random.shuffle(dataset['elements'])

            yield self.collate_fn(batch) if self.collate_fn is not None else batch


    def __len__(self):
        return int(self.dataset_lengths_sum / self.batch_size_pairs)


## SentenceTransformer model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

datasets = []
for filepath in sys.argv[1:]:
    dataset = []
    with gzip.open(filepath, 'rt', encoding='utf8') as fIn:
        for line in fIn:
            dataset.append(InputExample(texts=line.strip().split("\t")))

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
          use_amp=use_amp          #Set to True, if your GPU supports FP16 operations
          )
