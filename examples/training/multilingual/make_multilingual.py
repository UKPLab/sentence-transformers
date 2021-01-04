"""
This script contains an example how to extend an existent sentence embedding model to new languages.

Given a (monolingual) teacher model you would like to extend to new languages, which is specified in the teacher_model_name
variable. We train a multilingual student model to imitate the teacher model (variable student_model_name)
on multiple languages.

For training, you need parallel sentence data (machine translation training data). You need tab-seperated files (.tsv)
with the first column a sentence in a language understood by the teacher model, e.g. English,
and the further columns contain the according translations for languages you want to extend to.

This scripts downloads automatically the TED2020 corpus: https://github.com/UKPLab/sentence-transformers/blob/master/docs/datasets/TED2020.md
This corpus contains transcripts from
TED and TEDx talks, translated to 100+ languages. For other parallel data, see get_parallel_data_[].py scripts

Further information can be found in our paper:
Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation
https://arxiv.org/abs/2004.09813
"""

from sentence_transformers import SentenceTransformer, LoggingHandler, models, evaluation, losses
from torch.utils.data import DataLoader
from sentence_transformers.datasets import ParallelSentencesDataset
from datetime import datetime

import os
import logging
import sentence_transformers.util
import csv
import gzip
from tqdm.autonotebook import tqdm
import numpy as np
import zipfile
import io

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


teacher_model_name = 'bert-base-nli-stsb-mean-tokens'   #Our monolingual teacher model, we want to convert to multiple languages
student_model_name = 'xlm-roberta-base'                 #Multilingual base model we use to imitate the teacher model

max_seq_length = 128                #Student model max. lengths for inputs (number of word pieces)
train_batch_size = 64               #Batch size for training
inference_batch_size = 64           #Batch size at inference
max_sentences_per_language = 500000 #Maximum number of  parallel sentences for training
train_max_sentence_length = 250     #Maximum length (characters) for parallel training sentences

num_epochs = 5                       #Train for x epochs
num_warmup_steps = 10000             #Warumup steps

num_evaluation_steps = 1000          #Evaluate performance after every xxxx steps
dev_sentences = 1000                 #Number of parallel sentences to be used for development


# Define the language codes you would like to extend the model to
source_languages = set(['en'])                      # Our teacher model accepts English (en) sentences
target_languages = set(['de', 'es', 'it', 'fr', 'ar', 'tr'])    # We want to extend the model to these new languages. For language codes, see the header of the train file


output_path = "output/make-multilingual-"+"-".join(sorted(list(source_languages))+sorted(list(target_languages)))+"-"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# This function downloads a corpus if it does not exist
def download_corpora(filepaths):
    if not isinstance(filepaths, list):
        filepaths = [filepaths]

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(filepath, "does not exists. Try to download from server")
            filename = os.path.basename(filepath)
            url = "https://sbert.net/datasets/" + filename
            sentence_transformers.util.http_get(url, filepath)


# Here we define train train and dev corpora
train_corpus = "datasets/ted2020.tsv.gz"         # Transcripts of TED talks, crawled 2020
sts_corpus = "datasets/STS2017-extended.zip"     # Extended STS2017 dataset for more languages
parallel_sentences_folder = "parallel-sentences/"

# Check if the file exists. If not, they are downloaded
download_corpora([train_corpus, sts_corpus])


# Create parallel files for the selected language combinations
os.makedirs(parallel_sentences_folder, exist_ok=True)
train_files = []
dev_files = []
files_to_create = []
for source_lang in source_languages:
    for target_lang in target_languages:
        output_filename_train = os.path.join(parallel_sentences_folder, "TED2020-{}-{}-train.tsv.gz".format(source_lang, target_lang))
        output_filename_dev = os.path.join(parallel_sentences_folder, "TED2020-{}-{}-dev.tsv.gz".format(source_lang, target_lang))
        train_files.append(output_filename_train)
        dev_files.append(output_filename_dev)
        if not os.path.exists(output_filename_train) or not os.path.exists(output_filename_dev):
            files_to_create.append({'src_lang': source_lang, 'trg_lang': target_lang,
                                    'fTrain': gzip.open(output_filename_train, 'wt', encoding='utf8'),
                                    'fDev': gzip.open(output_filename_dev, 'wt', encoding='utf8'),
                                    'devCount': 0
                                    })

if len(files_to_create) > 0:
    print("Parallel sentences files {} do not exist. Create these files now".format(", ".join(map(lambda x: x['src_lang']+"-"+x['trg_lang'], files_to_create))))
    with gzip.open(train_corpus, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for line in tqdm(reader, desc="Sentences"):
            for outfile in files_to_create:
                src_text = line[outfile['src_lang']].strip()
                trg_text = line[outfile['trg_lang']].strip()

                if src_text != "" and trg_text != "":
                    if outfile['devCount'] < dev_sentences:
                        outfile['devCount'] += 1
                        fOut = outfile['fDev']
                    else:
                        fOut = outfile['fTrain']

                    fOut.write("{}\t{}\n".format(src_text, trg_text))

    for outfile in files_to_create:
        outfile['fTrain'].close()
        outfile['fDev'].close()



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
    train_data.load_data(train_file, max_sentences=max_sentences_per_language, max_sentence_length=train_max_sentence_length)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MSELoss(model=student_model)



#### Evaluate cross-lingual performance on different tasks #####
evaluators = []         #evaluators has a list of different evaluator classes we call periodically

for dev_file in dev_files:
    logger.info("Create evaluator for " + dev_file)
    src_sentences = []
    trg_sentences = []
    with gzip.open(dev_file, 'rt', encoding='utf8') as fIn:
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


##### Read cross-lingual Semantic Textual Similarity (STS) data ####
all_languages = list(set(list(source_languages)+list(target_languages)))
sts_data = {}

#Open the ZIP File of STS2017-extended.zip and check for which language combinations we have STS data
with zipfile.ZipFile(sts_corpus) as zip:
    filelist = zip.namelist()
    sts_files = []

    for i in range(len(all_languages)):
        for j in range(i, len(all_languages)):
            lang1 = all_languages[i]
            lang2 = all_languages[j]
            filepath = 'STS2017-extended/STS.{}-{}.txt'.format(lang1, lang2)
            if filepath not in filelist:
                lang1, lang2 = lang2, lang1
                filepath = 'STS2017-extended/STS.{}-{}.txt'.format(lang1, lang2)

            if filepath in filelist:
                filename = os.path.basename(filepath)
                sts_data[filename] = {'sentences1': [], 'sentences2': [], 'scores': []}

                fIn = zip.open(filepath)
                for line in io.TextIOWrapper(fIn, 'utf8'):
                    sent1, sent2, score = line.strip().split("\t")
                    score = float(score)
                    sts_data[filename]['sentences1'].append(sent1)
                    sts_data[filename]['sentences2'].append(sent2)
                    sts_data[filename]['scores'].append(score)

for filename, data in sts_data.items():
    test_evaluator = evaluation.EmbeddingSimilarityEvaluator(data['sentences1'], data['sentences2'], data['scores'], batch_size=inference_batch_size, name=filename, show_progress_bar=False)
    evaluators.append(test_evaluator)


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
