import torch
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer, LoggingHandler, models, util, InputExample
from sentence_transformers import losses
import os
import gzip
import csv
import random
import math
import logging

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

## Training parameters
model_name = 'bert-base-uncased'
batch_size = 16


################# Download and load STSb #################
data_folder = 'datasets/stsbenchmark'
sts_dataset_path = f'{data_folder}/stsbenchmark.tsv.gz'

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)


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

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')

#### Train sentences
# We use 1 Million sentences from Wikipedia to train our model
wikipedia_dataset_path = 'datasets/wiki1m_for_simcse.txt'
if not os.path.exists(wikipedia_dataset_path):
    util.http_get('https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt', wikipedia_dataset_path)

# train_sentences are simply your list of sentences
train_sentences = []
with open(wikipedia_dataset_path, 'r', encoding='utf8') as fIn:
    for line in fIn:
        train_sentences.append(line.strip())

################# Intialize an SBERT model #################


word_embedding_model = models.Transformer(model_name, max_seq_length=75)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

################# CT Data Loader #################
# For CT, we need batches in a specific format
# In each batch, we have one positive pair (i.e. [sentA, sentA]) and 7 negative pairs (i.e. [sentA, sentB]).
# To achieve this, we create a custom DataLoader that produces batches with this property

class ContrastiveTensionDataLoader:
    def __init__(self, sentences, batch_size, pos_neg_ratio=8):
        self.sentences = sentences
        self.batch_size = batch_size
        self.pos_neg_ratio = pos_neg_ratio
        self.collate_fn = None

        if self.batch_size % self.pos_neg_ratio != 0:
            raise ValueError(f"ContrastiveTensionDataLoader was loaded with a pos_neg_ratio of {pos_neg_ratio} and a batch size of {batch_size}. The batch size must be devisable by the pos_neg_ratio")

    def __iter__(self):
        random.shuffle(self.sentences)
        sentence_idx = 0
        batch = []

        while sentence_idx < len(self.sentences):
            s1 = self.sentences[sentence_idx]
            if len(batch) % self.pos_neg_ratio > 0:    #Negative (different) pair
                sentence_idx += 1
                s2 = self.sentences[sentence_idx]
                label = 0
            else:   #Positive (identical pair)
                s2 = self.sentences[sentence_idx]
                label = 1

            sentence_idx += 1
            batch.append(InputExample(texts=[s1, s2], label=label))

            if len(batch) >= self.batch_size:
                yield self.collate_fn(batch) if self.collate_fn is not None else batch
                batch = []

    def __len__(self):
        return math.floor(len(self.sentences)/self.batch_size)

# Use ContrastiveTensionReader and pass all training sentences we have
train_dataloader = ContrastiveTensionDataLoader(train_sentences, batch_size=batch_size)

# As loss, we losses.ContrastiveTensionLoss
train_loss = losses.ContrastiveTensionLoss(model)



"""
warmup_steps = 1000
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=1,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
    #weight_decay=0,
    output_path='result',
    use_amp=True
)
"""



model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=1,
    evaluation_steps=1000,
    weight_decay=0,
    warmup_steps=0,
    optimizer_class=torch.optim.RMSprop,
    optimizer_params={'lr': 1e-5},
    output_path='result-rmsprop',
    use_amp=False    #Set to True, if your GPU has optimized FP16 cores
)

###########

test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_score = model.evaluate(test_evaluator)
with open(os.path.join('result', 'test_score.txt'), 'w') as f:
    f.write(str(test_score) + '\n')