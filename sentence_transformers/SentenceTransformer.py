import os
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME, PretrainedConfig, BertConfig, XLNetConfig
import torch
from torch import Tensor
from numpy import ndarray
from typing import List, Tuple, Callable
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import importlib

from .util import http_get
from .config import SentenceTransformerConfig, LossFunction
from .evaluation import SentenceEvaluator
from .encoder import SentenceEncoder
from .trainer import SentenceTrainer, TrainConfig


class SentenceTransformer:
    """
    Wrapper around a Sentence BERT model for easy use.
    """
    def __init__(self, model_name_or_path: str = None, sentence_transformer_config: SentenceTransformerConfig = None):
        """
        Creates a Sentence BERT model based on either a pretrained model downloaded from the internet or the file system
        or based on a config for a new model

        When a model_url is given, then the files are downloaded from the URL. They are stored at model_path or in a
        temp folder based on the URL, when no model_path is given.
        When no model_url is given, but a model_path, then the model is loaded from the file system.
        When neither url nor path is given, then a new model is created based on the sbert_config

        :param model_name_or_path:
            A pre-trained model name, a URL or a path on the file system
        :param sentence_transformer_config:
            configuration for a new model
        """
        model_path = None
        if model_name_or_path is not None:
            if '/' not in model_name_or_path and '\\' not in model_name_or_path:
                model_name_or_path = 'https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.1/' + model_name_or_path

            if model_name_or_path.startswith('http://') or model_name_or_path.startswith('https://'):
                model_url = model_name_or_path
                folder_name = model_url.replace("https://", "").replace("http://", "").replace("/", "_")[:250]

                try:
                    from torch.hub import _get_torch_home
                    torch_cache_home = _get_torch_home()
                except ImportError:
                    torch_cache_home = os.path.expanduser(
                        os.getenv('TORCH_HOME', os.path.join(
                            os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
                default_cache_path = os.path.join(torch_cache_home, 'sentence_transformers')
                model_path = os.path.join(default_cache_path, folder_name)
                os.makedirs(model_path, exist_ok=True)

                if not os.listdir(model_path):
                    if model_url[-1] is "/":
                        model_url = model_url[:-1]
                    logging.info("Downloading sentence transformer model from {} and saving it at {}".format(model_url, model_path))
                    download_progress = tqdm(total=3, unit="files")
                    http_get(model_url + "/" + WEIGHTS_NAME, os.path.join(model_path, WEIGHTS_NAME))
                    download_progress.update(1)
                    http_get(model_url + "/" + CONFIG_NAME, os.path.join(model_path, CONFIG_NAME))
                    download_progress.update(1)
                    http_get(model_url + "/" + 'sentence_transformer_config.json', os.path.join(model_path, 'sentence_transformer_config.json'))
                    download_progress.update(1)
                    download_progress.close()
            else:
                model_path = model_name_or_path


        if model_path is not None:
            logging.info("Loading model from {}".format(model_path))
            output_model_file = os.path.join(model_path, WEIGHTS_NAME)
            output_transformer_config_file = os.path.join(model_path, CONFIG_NAME)
            output_sentence_transformer_config_file = os.path.join(model_path, 'sentence_transformer_config.json')

            sentence_transformer_config = SentenceTransformerConfig.from_json_file(output_sentence_transformer_config_file)
            logging.info("Transformer Model config {}".format(sentence_transformer_config))

            transformer_config = PretrainedConfig.from_json_file(output_transformer_config_file)
            model_class = self.import_from_string(sentence_transformer_config.model)
            self.transformer_model = model_class(transformer_config, sentence_transformer_config=sentence_transformer_config)
            self.transformer_model.load_state_dict(torch.load(output_model_file, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
            
        elif sentence_transformer_config is not None:
            logging.info("Creating a new {} model with config {}".format(sentence_transformer_config.model, sentence_transformer_config))
            model_class = self.import_from_string(sentence_transformer_config.model)
            self.transformer_model = model_class.from_pretrained(sentence_transformer_config.tokenizer_model)
            self.transformer_model.set_config(sentence_transformer_config)
            
        else:
            raise ValueError("model_url, model_path and config can not be all None.")

        self.transformer_model.set_tokenizer(sentence_transformer_config.tokenizer_model, sentence_transformer_config.do_lower_case)
        self.encoder = SentenceEncoder(self.transformer_model, sentence_transformer_config)
        self.trainer = SentenceTrainer(self.transformer_model)

    def import_from_string(self, dotted_path):
        """
        Import a dotted module path and return the attribute/class designated by the
        last name in the path. Raise ImportError if the import failed.
        """
        try:
            module_path, class_name = dotted_path.rsplit('.', 1)
        except ValueError:
            msg = "%s doesn't look like a module path" % dotted_path
            raise ImportError(msg)

        module = importlib.import_module(module_path)

        try:
            return getattr(module, class_name)
        except AttributeError:
            msg = 'Module "%s" does not define a "%s" attribute/class' % (module_path, class_name)
            raise ImportError(msg)

    def encode(self, sentences: List[str], batch_size: int = 32, show_progress_bar: bool = None) -> List[ndarray]:
        """
        Get the Sentence BERT embedding for a list of sentences

        :param sentences
            list of sentences to embed
        :param batch_size
            the batch size used for the encoding
        :return: a list of ndarrays with the embedding for each sentence
        """
        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)

        return self.encoder.get_sentence_embeddings(sentences, batch_size, show_progress_bar)

    def train(self, dataloader: DataLoader, train_config: TrainConfig):
        """
        Train the Sentence BERT model on the given data

        :param dataloader
            the data for the training as a DataLoader
        :param train_config
            the training configuration
        """
        self.trainer.train(dataloader, train_config)

    def multitask_train(self, dataloaders: List[DataLoader], losses: List[LossFunction], train_config: TrainConfig):
        """
        Train the model with the given data and config with the given loss for each dataset

        Each dataloader is sampled in turn for one batch.
        We sample only as many batches from each dataloader as there are in the smallest one
        to make sure of equal training with each dataset.

        :param dataloaders:
            the data for the training
        :param losses:
            the losses for each dataloader
            the losses still uses the configuration as given in sentence_transformer_config, so you cannot for example
            have two different LossFunction.SOFTMAX with different number of labels
        :param train_config:
            the configuration for the training
        """
        self.trainer.multitask_train(dataloaders, losses, train_config)

    def evaluate(self, evaluator: SentenceEvaluator, output_path: str = None):
        """"
        Evaluate the model

        :param evaluator:
            the evaluator
        :param output_path:
            the evaluator can write the results to this path
            The directories will be created, if they do not exist.
            Files in the folder might be overwritten or appended
        """
        self.trainer.evaluate(evaluator, output_path)

    def save(self, path: str):
        """
        Save the Sentence BERT model at the given path

        Directories are created if they do not exist yet.
        The model can be reloaded with SentenceTransformer(model_path=path).

        :param path:
            path where the model will be saved
            The directories will be created, if they do not exist.
            Previous models at the path will be overwritten
        """
        self.trainer.save(path)

    def smart_batching_collate(self) -> Callable[[List[Tuple[List[List[str]], Tensor]]],
                                                 Tuple[List[Tensor], List[Tensor], List[Tensor], Tensor]]:
        """
        Collate function to transform a batch from a SmartBatchingDataset to a batch of tensors for the model
        """
        return self.encoder.smart_batching_collate
