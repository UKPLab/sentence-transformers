"""
This script contains an example how to extend a model to new languages.

We use an existent (English) teacher sentence embedding model and extend it to a new language, in this case, German.

In order to run this example, you must download these files:
https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/TED2013-en-de.txt.gz
https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/STS2017.en-de.txt.gz
https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/xnli-en-de.txt.gz

And store them in the datasets-folder.

You can then run this code like this:
python training_multilingual.py datasets/TED2013-en-de.txt.gz
"""

from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models, readers, evaluation, losses
from torch.utils.data import DataLoader
from sentence_transformers.datasets import ParallelSentencesDataset
from datetime import datetime

import csv
import logging
import sys
import torch
import os
import numpy as np

#We can pass multiple train files to this script
train_files = sys.argv[1:]


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

max_seq_length = 128
train_batch_size = 64

logging.info("Load teacher model")
teacher_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

logging.info("Create student model from scratch")
word_embedding_model = models.XLMRoBERTa("xlm-roberta-base", do_lower_case=False)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

output_path = "output/make-multilingual-"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.info("Create dataset reader")


###### Read Dataset ######
train_data = ParallelSentencesDataset(student_model=model, teacher_model=teacher_model)
for train_file in train_files:
    train_data.load_data(train_file)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MSELoss(model=model)


###### Load dev sets ######

# Test on STS 2017.en-de dataset using Spearman rank correlation
logging.info("Read STS2017.en-de dataset")
evaluators = []
sts_reader = readers.STSDataReader('datasets/', s1_col_idx=0, s2_col_idx=1, score_col_idx=2)
dev_data = SentencesDataset(examples=sts_reader.get_examples('STS2017.en-de.txt.gz'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
evaluator_sts = evaluation.EmbeddingSimilarityEvaluator(dev_dataloader, name='STS2017.en-de')
evaluators.append(evaluator_sts)


# Use XLNI.en-de dataset with MSE evaluation
logging.info("Read XNLI.en-de dataset")
xnli_reader = ParallelSentencesDataset(student_model=model, teacher_model=teacher_model)
xnli_reader.load_data('datasets/xnli-en-de.txt.gz')

xnli_dataloader = DataLoader(xnli_reader, shuffle=False, batch_size=train_batch_size)
xnli_mse = evaluation.MSEEvaluator(xnli_dataloader, name='xnli-en-de')
evaluators.append(xnli_mse)



# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1]),
          epochs=20,
          evaluation_steps=5000,
          warmup_steps=10000,
          scheduler='warmupconstant',
          output_path=output_path,
          save_best_model=True,
          optimizer_params= {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False}
          )


