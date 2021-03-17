"""
This file contains an example how to make a SentenceTransformer model faster and lighter.

This is achieved by using Knowledge Distillation: We use a well working teacher model to train
a fast and light student model. The student model learns to imitate the produced
sentence embeddings from the teacher. We train this on a diverse set of sentences we got
from SNLI + Multi+NLI + Wikipedia.

After the distillation is finished, the student model produce nearly the same embeddings as the
teacher, however, it will be much faster.

The script implements to options two options to initialize the student:
Option 1: Train a light transformer model like TinyBERT to imitate the teacher
Option 2: We take the teacher model and keep only certain layers, for example, only 4 layers.

Option 2) works usually better, as we keep most of the weights from the teacher. In Option 1, we have to tune all
weights in the student from scratch.

There is a performance - speed trade-off. However, we found that a student with 4 instead of 12 layers keeps about 99.4%
of the teacher performance, while being 2.3 times faster.
"""
from torch.utils.data import DataLoader
from sentence_transformers import models, losses, evaluation
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.datasets import ParallelSentencesDataset
import logging
from datetime import datetime
import os
import gzip
import csv
import random
from sklearn.decomposition import PCA
import torch


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


# Teacher Model: Model we want to distill to a smaller model
teacher_model_name = 'stsb-roberta-base'
teacher_model = SentenceTransformer(teacher_model_name)

output_path = "output/model-distillation-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


use_layer_reduction = True

#There are two options to create a light and fast student model:
if use_layer_reduction:
    # 1) Create a smaller student model by using only some of the teacher layers
    student_model = SentenceTransformer(teacher_model_name)

    # Get the transformer model
    auto_model = student_model._first_module().auto_model

    # Which layers to keep from the teacher model. We equally spread the layers to keep over the original teacher
    #layers_to_keep = [5]
    #layers_to_keep = [3, 7]
    #layers_to_keep = [3, 7, 11]
    layers_to_keep = [1, 4, 7, 10]          #Keep 4 layers from the teacher
    #layers_to_keep = [0, 2, 4, 6, 8, 10]
    #layers_to_keep = [0, 1, 3, 4, 6, 7, 9, 10]

    logging.info("Remove layers from student. Only keep these layers: {}".format(layers_to_keep))
    new_layers = torch.nn.ModuleList([layer_module for i, layer_module in enumerate(auto_model.encoder.layer) if i in layers_to_keep])
    auto_model.encoder.layer = new_layers
    auto_model.config.num_hidden_layers = len(layers_to_keep)
else:
    # 2) The other option is to train a small model like TinyBERT to imitate the teacher.
    # You can find some small BERT models here: https://huggingface.co/nreimers
    word_embedding_model = models.Transformer('nreimers/TinyBERT_L-4_H-312_v2')
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    student_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])



inference_batch_size = 64
train_batch_size = 64



#We use AllNLI as a source of sentences for the distillation
nli_dataset_path = 'datasets/AllNLI.tsv.gz'

#Further, we use sentences extracted from the English Wikipedia to train the distillation
wikipedia_dataset_path = 'datasets/wikipedia-en-sentences.txt.gz'

#We use the STS benchmark dataset to see how much performance we loose
sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'


#Download datasets if needed
if not os.path.exists(nli_dataset_path):
    util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

if not os.path.exists(wikipedia_dataset_path):
    util.http_get('https://sbert.net/datasets/wikipedia-en-sentences.txt.gz', wikipedia_dataset_path)

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

#We need sentences to train our distillation. Here, we use sentences from AllNLI and from WikiPedia
train_sentences_nli = set()
dev_sentences_nli = set()

train_sentences_wikipedia = []
dev_sentences_wikipedia = []

# Read ALLNLI
with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'dev':
            dev_sentences_nli.add(row['sentence1'])
            dev_sentences_nli.add(row['sentence2'])
        else:
            train_sentences_nli.add(row['sentence1'])
            train_sentences_nli.add(row['sentence2'])

train_sentences_nli = list(train_sentences_nli)
random.shuffle(train_sentences_nli)

dev_sentences_nli = list(dev_sentences_nli)
random.shuffle(dev_sentences_nli)
dev_sentences_nli = dev_sentences_nli[0:5000] #Limit dev sentences to 5k

# Read Wikipedia sentences file
with gzip.open(wikipedia_dataset_path, 'rt', encoding='utf8') as fIn:
    wikipeda_sentences = [line.strip() for line in fIn]

dev_sentences_wikipedia = wikipeda_sentences[0:5000] #Use the first 5k sentences from the wikipedia file for development
train_sentences_wikipedia = wikipeda_sentences[5000:]


# We use the STS benchmark dataset to measure the performance of student model im comparison to the teacher model
logging.info("Read STSbenchmark dev dataset")
dev_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'dev':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

dev_evaluator_sts = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')


logging.info("Teacher Performance:")
dev_evaluator_sts(teacher_model)

# Student model has fewer dimensions. Compute PCA for the teacher to reduce the dimensions
if student_model.get_sentence_embedding_dimension() < teacher_model.get_sentence_embedding_dimension():
    logging.info("Student model has fewer dimensions than the teacher. Compute PCA for down projection")
    pca_sentences = train_sentences_nli[0:20000] + train_sentences_wikipedia[0:20000]
    pca_embeddings = teacher_model.encode(pca_sentences, convert_to_numpy=True)
    pca = PCA(n_components=student_model.get_sentence_embedding_dimension())
    pca.fit(pca_embeddings)

    #Add Dense layer to teacher that projects the embeddings down to the student embedding size
    dense = models.Dense(in_features=teacher_model.get_sentence_embedding_dimension(), out_features=student_model.get_sentence_embedding_dimension(), bias=False, activation_function=torch.nn.Identity())
    dense.linear.weight = torch.nn.Parameter(torch.tensor(pca.components_))
    teacher_model.add_module('dense', dense)

    logging.info("Teacher Performance with {} dimensions:".format(teacher_model.get_sentence_embedding_dimension()))
    dev_evaluator_sts(teacher_model)



# We train the student_model such that it creates sentence embeddings similar to the embeddings from the teacher_model
# For this, we need a large set of sentences. These sentences are embedded using the teacher model,
# and the student tries to mimic these embeddings. It is the same approach as used in: https://arxiv.org/abs/2004.09813
train_data = ParallelSentencesDataset(student_model=student_model, teacher_model=teacher_model, batch_size=inference_batch_size, use_embedding_cache=False)
train_data.add_dataset([[sent] for sent in train_sentences_nli], max_sentence_length=256)
train_data.add_dataset([[sent] for sent in train_sentences_wikipedia], max_sentence_length=256)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MSELoss(model=student_model)

# We create an evaluator, that measure the Mean Squared Error (MSE) between the teacher and the student embeddings
dev_sentences = dev_sentences_nli + dev_sentences_wikipedia
dev_evaluator_mse = evaluation.MSEEvaluator(dev_sentences, dev_sentences, teacher_model=teacher_model)

# Train the student model to imitate the teacher
student_model.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=evaluation.SequentialEvaluator([dev_evaluator_sts, dev_evaluator_mse]),
                  epochs=1,
                  warmup_steps=1000,
                  evaluation_steps=5000,
                  output_path=output_path,
                  save_best_model=True,
                  optimizer_params={'lr': 1e-4, 'eps': 1e-6, 'correct_bias': False},
                  use_amp=True)

