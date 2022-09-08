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

To run this example, you can use a shell script in which you can change the parameters and experiment. 
To do this, from the root of the repository in the terminal call the following command:
$ sh examples/training/multilingual/run_make_multilingual.sh

Further information can be found in our paper:
Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation
https://arxiv.org/abs/2004.09813
"""


from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import logging
import zipfile
import gzip
import sys
import csv
import os
import io

from sentence_transformers import (
    SentenceTransformer,
    LoggingHandler,
    models,
    evaluation,
    losses,
)
from sentence_transformers.datasets import ParallelSentencesDataset
from transformers import HfArgumentParser
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
import sentence_transformers.util
import numpy as np


@dataclass
class TrainingArguments:
    train_batch_size: int = field(
        default=64,
        metadata={
            "help": (
                "Batch size for training"
            )
        },
    )

    inference_batch_size: int = field(
        default=64,
        metadata={
            "help": (
                "Batch size at inference"
            )
        },
    )

    max_sentences_per_language: int = field(
        default=500000,
        metadata={
            "help": (
                "Maximum number of  parallel sentences for training"
            )
        },
    )

    train_max_sentence_length: int = field(
        default=250,
        metadata={
            "help": (
                "Maximum length (characters) for parallel training sentences"
            )
        },
    )

    num_epochs: int = field(
        default=5,
        metadata={
            "help": (
                "Number of model training epochs"
            )
        },
    )

    num_warmup_steps: int = field(
        default=10000,
        metadata={
            "help": (
                "Number of warm up steps"
            )
        },
    )

    num_evaluation_steps: int = field(
        default=1000,
        metadata={
            "help": (
                "Evaluate performance after every xxxx steps"
            )
        },
    )

    dev_sentences: int = field(
        default=1000,
        metadata={
            "help": (
                "Number of parallel sentences to be used for development"
            )
        },
    )

    output_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Storage path for the model and evaluation files"
            )
        }
    )

    learning_rate: float = field(
        default=2e-5,
        metadata={
            "help": (
                "The initial learning rate for optimizer"
            )
        }
    )

    epsilon: float = field(
        default=1e-6,
        metadata={
            "help": (
                "Epsilon for optimizer"
            )
        }
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which models we are going to use
    """
    teacher_model_name: str = field(
        metadata={
            "help": (
                "Our monolingual teacher model, we want to convert to multiple languages"
            )
        },
    )

    student_model_name: str = field(
        metadata={
            "help": (
                "Multilingual base model we use to imitate the teacher model"
            )
        },
    )

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "Student model max lengths for inputs (number of word pieces)"
            )
        },
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval
    """
    source_languages: list[str] = field(
        metadata={
            "help": (
                "Languages into which the teacher model accepts"
            )
        },
    )

    target_languages: list[str] = field(
        metadata={
            "help": (
                "The languages into which we want to extend the understanding of our model. \
                For language codes, see the header of the train file"
            )
        },
    )

    train_corpus: str = field(
        metadata={
            "help": (
                "Path to train corpus or name datasets from here https://sbert.net/datasets/"
            )
        },
    )

    val_corpus: str = field(
        metadata={
            "help": (
                "Path to validation corpus or name datasets from here https://sbert.net/datasets/"
            )
        },
    )

    path_to_parallel_folder: str = field(
        default='parallel-sentences/',
        metadata={
            "help": (
                "Path to folder where will be created files for the selected language combinations"
            )
        },
    )


    def __post_init__(self):
        if isinstance(self.source_languages, str):
            self.source_languages = [self.source_languages]
        if isinstance(self.target_languages, str):
            self.target_languages = [self.target_languages]

        self.source_languages = list(set(self.source_languages))
        self.target_languages = list(set(self.target_languages))


# Initializing logger global
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)


# This function downloads a corpus if it does not exist
def download_corpora(filepaths):
    if not isinstance(filepaths, list):
        filepaths = [filepaths]

    for filepath in filepaths:
        if not os.path.exists(filepath):
            logger.info(filepath, "does not exists. Try to download from server")
            filename = os.path.basename(filepath)
            url = "https://sbert.net/datasets/" + filename
            sentence_transformers.util.http_get(url, filepath)


# Create parallel files for the selected language combinations
def create_parallel_files(
    folder: os.PathLike,
    source_languages: list[str],
    target_languages: list[str],
    corpus: str,
    dev_sentences: int,
) -> tuple[list[os.PathLike], list[os.PathLike]]:

    os.makedirs(folder, exist_ok=True)
    train_files, dev_files, files_to_create = [], [], []

    for source_lang in source_languages:
        for target_lang in target_languages:
            output_filename_train = os.path.join(
                folder,
                f"TED2020-{source_lang}-{target_lang}-train.tsv.gz",
            )
            output_filename_dev = os.path.join(
                folder,
                f"TED2020-{source_lang}-{target_lang}-dev.tsv.gz",
            )
            train_files.append(output_filename_train)
            dev_files.append(output_filename_dev)
            if not (os.path.exists(output_filename_train) and
                    os.path.exists(output_filename_dev)):
                files_to_create.append({
                    'src_lang': source_lang,
                    'trg_lang': target_lang,
                    'fTrain': gzip.open(output_filename_train, 'wt', encoding='utf8'),
                    'fDev': gzip.open(output_filename_dev, 'wt', encoding='utf8'),
                    'devCount': 0,
                })

    if files_to_create:
        lang_pairs = ", ".join(
            map(lambda x: x['src_lang'] + "-" + x['trg_lang'], files_to_create)
        )
        logger.info(f"Parallel sentences files {lang_pairs} do not exist. Create these files now")
        with gzip.open(corpus, 'rt', encoding='utf8') as file_lines:
            reader = csv.DictReader(file_lines, delimiter='\t', quoting=csv.QUOTE_NONE)
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

                        fOut.write(f"{src_text}\t{trg_text}\n")

        for outfile in files_to_create:
            outfile['fTrain'].close()
            outfile['fDev'].close()
    
    return train_files, dev_files


def train_parallel_files_to_dataloader(
    train_files: list[os.PathLike],
    inference_batch_size: int,
    train_batch_size: int,
    max_sentences_per_language: int,
    train_max_sentence_length: int,
    student_model: SentenceTransformer,
    teacher_model: SentenceTransformer,
) -> DataLoader:

    train_data = ParallelSentencesDataset(
        student_model=student_model,
        teacher_model=teacher_model,
        batch_size=inference_batch_size,
        use_embedding_cache=True,
    )

    for train_file in train_files:
        train_data.load_data(
            train_file,
            max_sentences=max_sentences_per_language,
            max_sentence_length=train_max_sentence_length,
        )

    return DataLoader(
        train_data,
        shuffle=True,
        batch_size=train_batch_size,
    )


def create_diffirent_evalutors_for_validation(
    dev_files: list[os.PathLike],
    source_languages: list[str],
    target_languages: list[str],
    inference_batch_size: int,
    val_corpus: str,
    teacher_model: SentenceTransformer,
) -> list[evaluation.SentenceEvaluator]:
    # Evaluators has a list of different evaluator classes we call periodically
    evaluators = []

    for dev_file in dev_files:
        logger.info("Create evaluator for " + dev_file)
        src_sentences, trg_sentences = [], []
        with gzip.open(dev_file, 'rt', encoding='utf8') as file_lines:
            for line in file_lines:
                splits = line.strip().split('\t')
                if splits[0] != "" and splits[1] != "":
                    src_sentences.append(splits[0])
                    trg_sentences.append(splits[1])

        # Mean Squared Error (MSE) measures the (euclidean) distance between teacher and student embeddings
        dev_mse = evaluation.MSEEvaluator(
            src_sentences,
            trg_sentences,
            name=os.path.basename(dev_file),
            teacher_model=teacher_model,
            batch_size=inference_batch_size,
        )
        evaluators.append(dev_mse)

        # TranslationEvaluator computes the embeddings for all parallel sentences. It then check if the embedding of source[i] is the closest to target[i] out of all available target sentences
        dev_trans_acc = evaluation.TranslationEvaluator(
            src_sentences,
            trg_sentences,
            name=os.path.basename(dev_file),
            batch_size=inference_batch_size,
        )
        evaluators.append(dev_trans_acc)

    ##### Read cross-lingual Semantic Textual Similarity (STS) data ####
    all_languages = list(set(source_languages) | set(target_languages))
    sts_data = {}

    # Open the ZIP File of STS2017-extended.zip and check for which language combinations we have STS data
    with zipfile.ZipFile(val_corpus) as zip:
        filelist = zip.namelist()

        for i, lang1 in enumerate(all_languages):
            for lang2 in all_languages[i + 1:]:

                filepath = f'STS2017-extended/STS.{lang1}-{lang2}.txt'
                if filepath not in filelist:
                    lang1, lang2 = lang2, lang1
                    filepath = f'STS2017-extended/STS.{lang1}-{lang2}.txt'

                if filepath in filelist:
                    filename = os.path.basename(filepath)
                    sts_data[filename] = {
                        'sentences1': [],
                        'sentences2': [],
                        'scores': [],
                    }

                    file_bytes = zip.open(filepath)
                    for line in io.TextIOWrapper(file_bytes, 'utf8'):
                        sent1, sent2, score = line.strip().split("\t")
                        score = float(score)
                        sts_data[filename]['sentences1'].append(sent1)
                        sts_data[filename]['sentences2'].append(sent2)
                        sts_data[filename]['scores'].append(score)


    for filename, data in sts_data.items():
        test_evaluator = evaluation.EmbeddingSimilarityEvaluator(
            data['sentences1'],
            data['sentences2'],
            data['scores'],
            batch_size=inference_batch_size,
            name=filename,
            show_progress_bar=False,
        )
        evaluators.append(test_evaluator)
    
    return evaluators


def main():
    parser = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        train_args, model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        train_args, model_args, data_args = parser.parse_args_into_dataclasses()

    if not train_args.output_path:
        train_args.output_path = (
            "output/make-multilingual-"
            + "-".join(
                sorted(list(data_args.source_languages))
                + sorted(list(data_args.target_languages))
            )
            + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )

    logger.info(train_args)
    logger.info(model_args)
    logger.info(data_args)

    # Check if the file exists. If not, they are downloaded
    download_corpora([data_args.train_corpus, data_args.val_corpus])

    # Create parallel files for the selected language combinations
    train_files, dev_files = create_parallel_files(
        data_args.path_to_parallel_folder,
        data_args.source_languages,
        data_args.target_languages,
        data_args.train_corpus,
        train_args.dev_sentences,
    )

    ######## Start the extension of the teacher model to multiple languages ########
    logger.info("Load teacher model")
    teacher_model = SentenceTransformer(model_args.teacher_model_name)

    logger.info("Create student model from scratch")
    word_embedding_model = models.Transformer(
        model_args.student_model_name,
        max_seq_length=model_args.max_seq_length,
    )
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
    )
    student_model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model],
    )

    ###### Read Parallel Sentences Dataset ######
    train_dataloader = train_parallel_files_to_dataloader(
        train_files,
        train_args.inference_batch_size,
        train_args.train_batch_size,
        train_args.max_sentences_per_language,
        train_args.train_max_sentence_length,
        student_model,
        teacher_model,
    )

    train_loss = losses.MSELoss(model=student_model)

    #### Evaluate cross-lingual performance on different tasks #####
    evaluators = create_diffirent_evalutors_for_validation(
        dev_files,
        data_args.source_languages,
        data_args.target_languages,
        train_args.inference_batch_size,
        data_args.val_corpus,
        teacher_model,
    )

    # Train the model
    student_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluation.SequentialEvaluator(
            evaluators,
            main_score_function=lambda scores: np.mean(scores),
        ),
        epochs=train_args.num_epochs,
        warmup_steps=train_args.num_warmup_steps,
        evaluation_steps=train_args.num_evaluation_steps,
        output_path=train_args.output_path,
        save_best_model=True,
        optimizer_params={
            'lr': train_args.learning_rate,
            'eps': train_args.epsilon,
        },
    )


if __name__ == '__main__':
    main()
