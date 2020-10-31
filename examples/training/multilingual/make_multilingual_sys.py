"""
This script contains an example how to extend an existent sentence embedding model to new languages.

Given a (monolingual) teacher model you would like to extend to new languages, which is specified in the teacher_model_name
variable. We train a multilingual student model to imitate the teacher model (variable student_model_name)
on multiple languages.

For training, you need parallel sentence data (machine translation training data). You need tab-seperated files (.tsv)
with the first column a sentence in a language understood by the teacher model, e.g. English,
and the further columns contain the according translations for languages you want to extend to.

See get_parallel_data_[opus/tatoeba/ted2020].py for automatic download of parallel sentences datasets.

Note: See make_multilingual.py for a fully automated script that downloads the necessary data and trains the model. This script just trains the model if you have already parallel data in the right format.


Further information can be found in our paper:
Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation
https://arxiv.org/abs/2004.09813


Usage:
python make_multilingual_sys.py train1.tsv.gz train2.tsv.gz train3.tsv.gz --dev dev1.tsv.gz dev2.tsv.gz

For example:
python make_multilingual_sys.py parallel-sentences/TED2020-en-de-train.tsv.gz --dev parallel-sentences/TED2020-en-de-dev.tsv.gz

To load all training & dev files from a folder (Linux):
python make_multilingual_sys.py parallel-sentences/*-train.tsv.gz --dev parallel-sentences/*-dev.tsv.gz



"""

from sentence_transformers import SentenceTransformer, LoggingHandler, models, evaluation, losses
from torch.utils.data import DataLoader
from sentence_transformers.datasets import ParallelSentencesDataset
from datetime import datetime

import os
import logging
import gzip
import numpy as np
import sys

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


teacher_model_name = 'bert-base-nli-stsb-mean-tokens'   #Our monolingual teacher model, we want to convert to multiple languages
student_model_name = 'xlm-roberta-base'                 #Multilingual base model we use to imitate the teacher model

max_seq_length = 128                 #Student model max. lengths for inputs (number of word pieces)
train_batch_size = 64                #Batch size for training
inference_batch_size = 64            #Batch size at inference
max_sentences_per_trainfile = 500000 #Maximum number of  parallel sentences for training
train_max_sentence_length = 250      #Maximum length (characters) for parallel training sentences

num_epochs = 5                       #Train for x epochs
num_warmup_steps = 10000             #Warumup steps

num_evaluation_steps = 1000          #Evaluate performance after every xxxx steps



output_path = "output/make-multilingual-sys-"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


#Read passed arguments


train_files = []
dev_files = []
is_dev_file = False
for arg in sys.argv[1:]:
    if arg.lower() == '--dev':
        is_dev_file = True
    else:
        if not os.path.exists(arg):
            print("File could not be found:", arg)
            exit()

        if is_dev_file:
            dev_files.append(arg)
        else:
            train_files.append(arg)

if len(train_files) == 0:
    print("Please pass at least some train files")
    print("python make_multilingual_sys.py file1.tsv.gz file2.tsv.gz --dev dev1.tsv.gz dev2.tsv.gz")
    exit()


logger.info("Train files: {}".format(", ".join(train_files)))
logger.info("Dev files: {}".format(", ".join(dev_files)))

######## Start the extension of the teacher model to multiple languages ########
logger.info("Load teacher model")
teacher_model = SentenceTransformer(teacher_model_name)


logger.info("Create student model from scratch")
word_embedding_model = models.Transformer(student_model_name, max_seq_length=max_seq_length)
# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
student_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


###### Read Parallel Sentences Dataset ######
train_data = ParallelSentencesDataset(student_model=student_model, teacher_model=teacher_model, batch_size=inference_batch_size, use_embedding_cache=True)
for train_file in train_files:
    train_data.load_data(train_file, max_sentences=max_sentences_per_trainfile, max_sentence_length=train_max_sentence_length)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MSELoss(model=student_model)



#### Evaluate cross-lingual performance on different tasks #####
evaluators = []         #evaluators has a list of different evaluator classes we call periodically

for dev_file in dev_files:
    logger.info("Create evaluator for " + dev_file)
    src_sentences = []
    trg_sentences = []
    with gzip.open(dev_file, 'rt', encoding='utf8') if dev_file.endswith('.gz') else open(dev_file, encoding='utf8') as fIn:
        for line in fIn:
            splits = line.strip().split('\t')
            if splits[0] != "" and splits[1] != "":
                src_sentences.append(splits[0])
                trg_sentences.append(splits[1])


    #Mean Squared Error (MSE) measures the (euclidean) distance between teacher and student embeddings
    dev_mse = evaluation.MSEEvaluator(src_sentences, trg_sentences, name=os.path.basename(dev_file), teacher_model=teacher_model, batch_size=inference_batch_size)
    evaluators.append(dev_mse)

    # TranslationEvaluator computes the embeddings for all parallel sentences. It then check if the embedding of source[i] is the closest to target[i] out of all available target sentences
    dev_trans_acc = evaluation.TranslationEvaluator(src_sentences, trg_sentences, name=os.path.basename(dev_file),batch_size=inference_batch_size)
    evaluators.append(dev_trans_acc)



# Train the model
student_model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: np.mean(scores)),
          epochs=num_epochs,
          warmup_steps=num_warmup_steps,
          evaluation_steps=num_evaluation_steps,
          output_path=output_path,
          save_best_model=True,
          optimizer_params= {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False}
          )
