import json
import logging
import os
import shutil
import stat
from collections import OrderedDict
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
import requests
import numpy as np
from numpy import ndarray
import transformers
from huggingface_hub import HfApi, HfFolder, Repository, hf_hub_url, cached_download
import torch
from torch import nn, Tensor, device
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm.autonotebook import trange
import math
import queue
import tempfile
from distutils.dir_util import copy_tree

from . import __MODEL_HUB_ORGANIZATION__
from .evaluation import SentenceEvaluator
from .util import import_from_string, batch_to_device, fullname, snapshot_download
from .models import Transformer, Pooling, Dense
from .model_card_templates import ModelCardTemplate
from . import __version__

logger = logging.getLogger(__name__)

class SentenceTransformer(nn.Sequential):
    """
    Loads or create a SentenceTransformer model, that can be used to map sentences / text to embeddings.

    :param model_name_or_path: If it is a filepath on disc, it loads the model from that path. If it is not a path, it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model from Huggingface models repository with that name.
    :param modules: This parameter can be used to create custom SentenceTransformer models from scratch.
    :param device: Device (like 'cuda' / 'cpu') that should be used for computation. If None, checks if a GPU can be used.
    :param cache_folder: Path to store models. Can be also set by SENTENCE_TRANSFORMERS_HOME enviroment variable.
    :param use_auth_token: HuggingFace authentication token to download private models.
    """
    def __init__(self, model_name_or_path: Optional[str] = None,
                 modules: Optional[Iterable[nn.Module]] = None,
                 device: Optional[str] = None,
                 cache_folder: Optional[str] = None,
                 use_auth_token: Union[bool, str, None] = None
                 ):
        self._model_card_vars = {}
        self._model_card_text = None
        self._model_config = {}

        if cache_folder is None:
            cache_folder = os.getenv('SENTENCE_TRANSFORMERS_HOME')
            if cache_folder is None:
                try:
                    from torch.hub import _get_torch_home

                    torch_cache_home = _get_torch_home()
                except ImportError:
                    torch_cache_home = os.path.expanduser(os.getenv('TORCH_HOME', os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))

                cache_folder = os.path.join(torch_cache_home, 'sentence_transformers')

        if model_name_or_path is not None and model_name_or_path != "":
            logger.info("Load pretrained SentenceTransformer: {}".format(model_name_or_path))

            #Old models that don't belong to any organization
            basic_transformer_models = ['albert-base-v1', 'albert-base-v2', 'albert-large-v1', 'albert-large-v2', 'albert-xlarge-v1', 'albert-xlarge-v2', 'albert-xxlarge-v1', 'albert-xxlarge-v2', 'bert-base-cased-finetuned-mrpc', 'bert-base-cased', 'bert-base-chinese', 'bert-base-german-cased', 'bert-base-german-dbmdz-cased', 'bert-base-german-dbmdz-uncased', 'bert-base-multilingual-cased', 'bert-base-multilingual-uncased', 'bert-base-uncased', 'bert-large-cased-whole-word-masking-finetuned-squad', 'bert-large-cased-whole-word-masking', 'bert-large-cased', 'bert-large-uncased-whole-word-masking-finetuned-squad', 'bert-large-uncased-whole-word-masking', 'bert-large-uncased', 'camembert-base', 'ctrl', 'distilbert-base-cased-distilled-squad', 'distilbert-base-cased', 'distilbert-base-german-cased', 'distilbert-base-multilingual-cased', 'distilbert-base-uncased-distilled-squad', 'distilbert-base-uncased-finetuned-sst-2-english', 'distilbert-base-uncased', 'distilgpt2', 'distilroberta-base', 'gpt2-large', 'gpt2-medium', 'gpt2-xl', 'gpt2', 'openai-gpt', 'roberta-base-openai-detector', 'roberta-base', 'roberta-large-mnli', 'roberta-large-openai-detector', 'roberta-large', 't5-11b', 't5-3b', 't5-base', 't5-large', 't5-small', 'transfo-xl-wt103', 'xlm-clm-ende-1024', 'xlm-clm-enfr-1024', 'xlm-mlm-100-1280', 'xlm-mlm-17-1280', 'xlm-mlm-en-2048', 'xlm-mlm-ende-1024', 'xlm-mlm-enfr-1024', 'xlm-mlm-enro-1024', 'xlm-mlm-tlm-xnli15-1024', 'xlm-mlm-xnli15-1024', 'xlm-roberta-base', 'xlm-roberta-large-finetuned-conll02-dutch', 'xlm-roberta-large-finetuned-conll02-spanish', 'xlm-roberta-large-finetuned-conll03-english', 'xlm-roberta-large-finetuned-conll03-german', 'xlm-roberta-large', 'xlnet-base-cased', 'xlnet-large-cased']

            if os.path.exists(model_name_or_path):
                #Load from path
                model_path = model_name_or_path
            else:
                #Not a path, load from hub
                if '\\' in model_name_or_path or model_name_or_path.count('/') > 1:
                    raise ValueError("Path {} not found".format(model_name_or_path))

                if '/' not in model_name_or_path and model_name_or_path.lower() not in basic_transformer_models:
                    # A model from sentence-transformers
                    model_name_or_path = __MODEL_HUB_ORGANIZATION__ + "/" + model_name_or_path

                model_path = os.path.join(cache_folder, model_name_or_path.replace("/", "_"))
                
                if not os.path.exists(os.path.join(model_path, 'modules.json')):
                    # Download from hub with caching
                    snapshot_download(model_name_or_path,
                                        cache_dir=cache_folder,
                                        library_name='sentence-transformers',
                                        library_version=__version__,
                                        ignore_files=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5'],
                                        use_auth_token=use_auth_token)

            if os.path.exists(os.path.join(model_path, 'modules.json')):    #Load as SentenceTransformer model
                modules = self._load_sbert_model(model_path)
            else:   #Load with AutoModel
                modules = self._load_auto_model(model_path)

        if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])

        super().__init__(modules)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)



    def encode(self, sentences: Union[str, List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel()==logging.INFO or logger.getEffectiveLevel()==logging.DEBUG)

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != 'sentence_embedding':
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'): #Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = self._target_device

        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            features = self.tokenize(sentences_batch)
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.forward(features)

                if output_value == 'token_embeddings':
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features['attention_mask']):
                        last_mask_id = len(attention)-1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0:last_mask_id+1])
                elif output_value is None:  #Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features['sentence_embedding'])):
                        row =  {name: out_features[name][sent_idx] for name in out_features}
                        embeddings.append(row)
                else:   #Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings



    def start_multi_process_pool(self, target_devices: List[str] = None):
        """
        Starts multi process to process the encoding with several, independent processes.
        This method is recommended if you want to encode on multiple GPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process

        :param target_devices: PyTorch target devices, e.g. cuda:0, cuda:1... If None, all available CUDA devices will be used
        :return: Returns a dict with the target processes, an input queue and and output queue.
        """
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
            else:
                logger.info("CUDA is not available. Start 4 CPU worker")
                target_devices = ['cpu']*4

        logger.info("Start multi-process pool on devices: {}".format(', '.join(map(str, target_devices))))

        ctx = mp.get_context('spawn')
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for cuda_id in target_devices:
            p = ctx.Process(target=SentenceTransformer._encode_multi_process_worker, args=(cuda_id, self, input_queue, output_queue), daemon=True)
            p.start()
            processes.append(p)

        return {'input': input_queue, 'output': output_queue, 'processes': processes}


    @staticmethod
    def stop_multi_process_pool(pool):
        """
        Stops all processes started with start_multi_process_pool
        """
        for p in pool['processes']:
            p.terminate()

        for p in pool['processes']:
            p.join()
            p.close()

        pool['input'].close()
        pool['output'].close()


    def encode_multi_process(self, sentences: List[str], pool: Dict[str, object], batch_size: int = 32, chunk_size: int = None):
        """
        This method allows to run encode() on multiple GPUs. The sentences are chunked into smaller packages
        and sent to individual processes, which encode these on the different GPUs. This method is only suitable
        for encoding large sets of sentences

        :param sentences: List of sentences
        :param pool: A pool of workers started with SentenceTransformer.start_multi_process_pool
        :param batch_size: Encode sentences with batch size
        :param chunk_size: Sentences are chunked and sent to the individual processes. If none, it determine a sensible size.
        :return: Numpy matrix with all embeddings
        """

        if chunk_size is None:
            chunk_size = min(math.ceil(len(sentences) / len(pool["processes"]) / 10), 5000)

        logger.debug(f"Chunk data into {math.ceil(len(sentences) / chunk_size)} packages of size {chunk_size}")

        input_queue = pool['input']
        last_chunk_id = 0
        chunk = []

        for sentence in sentences:
            chunk.append(sentence)
            if len(chunk) >= chunk_size:
                input_queue.put([last_chunk_id, batch_size, chunk])
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, batch_size, chunk])
            last_chunk_id += 1

        output_queue = pool['output']
        results_list = sorted([output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0])
        embeddings = np.concatenate([result[1] for result in results_list])
        return embeddings

    @staticmethod
    def _encode_multi_process_worker(target_device: str, model, input_queue, results_queue):
        """
        Internal working process to encode sentences in multi-process setup
        """
        while True:
            try:
                id, batch_size, sentences = input_queue.get()
                embeddings = model.encode(sentences, device=target_device,  show_progress_bar=False, convert_to_numpy=True, batch_size=batch_size)
                results_queue.put([id, embeddings])
            except queue.Empty:
                break



    def get_max_seq_length(self):
        """
        Returns the maximal sequence length for input the model accepts. Longer inputs will be truncated
        """
        if hasattr(self._first_module(), 'max_seq_length'):
            return self._first_module().max_seq_length

        return None

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes the texts
        """
        return self._first_module().tokenize(texts)

    def get_sentence_features(self, *features):
        return self._first_module().get_sentence_features(*features)

    def get_sentence_embedding_dimension(self):
        for mod in reversed(self._modules.values()):
            sent_embedding_dim_method = getattr(mod, "get_sentence_embedding_dimension", None)
            if callable(sent_embedding_dim_method):
                return sent_embedding_dim_method()
        return None

    def _first_module(self):
        """Returns the first module of this sequential embedder"""
        return self._modules[next(iter(self._modules))]

    def _last_module(self):
        """Returns the last module of this sequential embedder"""
        return self._modules[next(reversed(self._modules))]

    def save(self, path: str, model_name: Optional[str] = None, create_model_card: bool = True, train_datasets: Optional[List[str]] = None):
        """
        Saves all elements for this seq. sentence embedder into different sub-folders
        :param path: Path on disc
        :param model_name: Optional model name
        :param create_model_card: If True, create a README.md with basic information about this model
        :param train_datasets: Optional list with the names of the datasets used to to train the model
        """
        if path is None:
            return

        os.makedirs(path, exist_ok=True)

        logger.info("Save model to {}".format(path))
        modules_config = []

        #Save some model info
        if '__version__' not in self._model_config:
            self._model_config['__version__'] = {
                    'sentence_transformers': __version__,
                    'transformers': transformers.__version__,
                    'pytorch': torch.__version__,
                }

        with open(os.path.join(path, 'config_sentence_transformers.json'), 'w') as fOut:
            json.dump(self._model_config, fOut, indent=2)

        #Save modules
        for idx, name in enumerate(self._modules):
            module = self._modules[name]
            if idx == 0 and isinstance(module, Transformer):    #Save transformer model in the main folder
                model_path = path + "/"
            else:
                model_path = os.path.join(path, str(idx)+"_"+type(module).__name__)

            os.makedirs(model_path, exist_ok=True)
            module.save(model_path)
            modules_config.append({'idx': idx, 'name': name, 'path': os.path.basename(model_path), 'type': type(module).__module__})

        with open(os.path.join(path, 'modules.json'), 'w') as fOut:
            json.dump(modules_config, fOut, indent=2)

        # Create model card
        if create_model_card:
            self._create_model_card(path, model_name, train_datasets)

    def _create_model_card(self, path: str, model_name: Optional[str] = None, train_datasets: Optional[List[str]] = None):
        """
        Create an automatic model and stores it in path
        """
        if self._model_card_text is not None and len(self._model_card_text) > 0:
            model_card = self._model_card_text
        else:
            tags = ModelCardTemplate.__TAGS__.copy()
            model_card = ModelCardTemplate.__MODEL_CARD__

            if len(self._modules) == 2 and isinstance(self._first_module(), Transformer) and isinstance(self._last_module(), Pooling) and self._last_module().get_pooling_mode_str() in ['cls', 'max', 'mean']:
                pooling_module = self._last_module()
                pooling_mode = pooling_module.get_pooling_mode_str()
                model_card = model_card.replace("{USAGE_TRANSFORMERS_SECTION}", ModelCardTemplate.__USAGE_TRANSFORMERS__)
                pooling_fct_name, pooling_fct = ModelCardTemplate.model_card_get_pooling_function(pooling_mode)
                model_card = model_card.replace("{POOLING_FUNCTION}", pooling_fct).replace("{POOLING_FUNCTION_NAME}", pooling_fct_name).replace("{POOLING_MODE}", pooling_mode)
                tags.append('transformers')

            # Print full model
            model_card = model_card.replace("{FULL_MODEL_STR}", str(self))

            # Add tags
            model_card = model_card.replace("{TAGS}", "\n".join(["- "+t for t in tags]))

            datasets_str = ""
            if train_datasets is not None:
                datasets_str = "datasets:\n"+"\n".join(["- " + d for d in train_datasets])
            model_card = model_card.replace("{DATASETS}", datasets_str)


            # Add dim info
            self._model_card_vars["{NUM_DIMENSIONS}"] = self.get_sentence_embedding_dimension()

            # Replace vars we created while using the model
            for name, value in self._model_card_vars.items():
                model_card = model_card.replace(name, str(value))

            # Replace remaining vars with default values
            for name, value in ModelCardTemplate.__DEFAULT_VARS__.items():
                model_card = model_card.replace(name, str(value))

        if model_name is not None:
            model_card = model_card.replace("{MODEL_NAME}", model_name.strip())

        with open(os.path.join(path, "README.md"), "w", encoding='utf8') as fOut:
            fOut.write(model_card.strip())

    def save_to_hub(self,
                    repo_name: str,
                    organization: Optional[str] = None,
                    private: Optional[bool] = None,
                    commit_message: str = "Add new SentenceTransformer model.",
                    local_model_path: Optional[str] = None,
                    exist_ok: bool = False,
                    replace_model_card: bool = False,
                    train_datasets: Optional[List[str]] = None):
        """
        Uploads all elements of this Sentence Transformer to a new HuggingFace Hub repository.

        :param repo_name: Repository name for your model in the Hub.
        :param organization:  Organization in which you want to push your model or tokenizer (you must be a member of this organization).
        :param private: Set to true, for hosting a prive model
        :param commit_message: Message to commit while pushing.
        :param local_model_path: Path of the model locally. If set, this file path will be uploaded. Otherwise, the current model will be uploaded
        :param exist_ok: If true, saving to an existing repository is OK. If false, saving only to a new repository is possible
        :param replace_model_card: If true, replace an existing model card in the hub with the automatically created model card
        :param train_datasets: Datasets used to train the model. If set, the datasets will be added to the model card in the Hub.
        :return: The url of the commit of your model in the given repository.
        """
        token = HfFolder.get_token()
        if token is None:
            raise ValueError("You must login to the Hugging Face hub on this computer by typing `transformers-cli login`.")

        if '/' in repo_name:
            splits = repo_name.split('/', maxsplit=1)
            if organization is None or organization == splits[0]:
                organization = splits[0]
                repo_name = splits[1]
            else:
                raise ValueError("You passed and invalid repository name: {}.".format(repo_name))

        endpoint = "https://huggingface.co"
        repo_id = repo_name
        if organization:
          repo_id = f"{organization}/{repo_id}"
        repo_url = HfApi(endpoint=endpoint).create_repo(
                repo_id=repo_id,
                token=token,
                private=private,
                repo_type=None,
                exist_ok=exist_ok,
            )
        full_model_name = repo_url[len(endpoint)+1:].strip("/")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # First create the repo (and clone its content if it's nonempty).
            logger.info("Create repository and clone it if it exists")
            repo = Repository(tmp_dir, clone_from=repo_url)

            # If user provides local files, copy them.
            if local_model_path:
                copy_tree(local_model_path, tmp_dir)
            else:  # Else, save model directly into local repo.
                create_model_card = replace_model_card or not os.path.exists(os.path.join(tmp_dir, 'README.md'))
                self.save(tmp_dir, model_name=full_model_name, create_model_card=create_model_card, train_datasets=train_datasets)

            #Find files larger 5M and track with git-lfs
            large_files = []
            for root, dirs, files in os.walk(tmp_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, tmp_dir)

                    if os.path.getsize(file_path) > (5 * 1024 * 1024):
                        large_files.append(rel_path)

            if len(large_files) > 0:
                logger.info("Track files with git lfs: {}".format(", ".join(large_files)))
                repo.lfs_track(large_files)

            logger.info("Push model to the hub. This might take a while")
            push_return = repo.push_to_hub(commit_message=commit_message)

            def on_rm_error(func, path, exc_info):
                # path contains the path of the file that couldn't be removed
                # let's just assume that it's read-only and unlink it.
                try:
                    os.chmod(path, stat.S_IWRITE)
                    os.unlink(path)
                except:
                    pass

            # Remove .git folder. On Windows, the .git folder might be read-only and cannot be deleted
            # Hence, try to set write permissions on error
            try:
                for f in os.listdir(tmp_dir):
                    shutil.rmtree(os.path.join(tmp_dir, f), onerror=on_rm_error)
            except Exception as e:
                logger.warning("Error when deleting temp folder: {}".format(str(e)))
                pass


        return push_return

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)

            labels.append(example.label)

        labels = torch.tensor(labels)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.tokenize(texts[idx])
            sentence_features.append(tokenized)

        return sentence_features, labels


    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):              #{key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):      #Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):    #Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])      #Sum of length of individual strings

    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            steps_per_epoch = None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            optimizer_params : Dict[str, object]= {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = 500,
            checkpoint_save_total_limit: int = 0
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        :param checkpoint_path: Folder to save checkpoints during training
        :param checkpoint_save_steps: Will save a checkpoint after so many steps
        :param checkpoint_save_total_limit: Total number of checkpoints to store
        """

        ##Add info to model card
        #info_loss_functions = "\n".join(["- {} with {} training examples".format(str(loss), len(dataloader)) for dataloader, loss in train_objectives])
        info_loss_functions =  []
        for dataloader, loss in train_objectives:
            info_loss_functions.extend(ModelCardTemplate.get_train_objective_info(dataloader, loss))
        info_loss_functions = "\n\n".join([text for text in info_loss_functions])

        info_fit_parameters = json.dumps({"evaluator": fullname(evaluator), "epochs": epochs, "steps_per_epoch": steps_per_epoch, "scheduler": scheduler, "warmup_steps": warmup_steps, "optimizer_class": str(optimizer_class),  "optimizer_params": optimizer_params, "weight_decay": weight_decay, "evaluation_steps": evaluation_steps, "max_grad_norm": max_grad_norm }, indent=4, sort_keys=True)
        self._model_card_text = None
        self._model_card_vars['{TRAINING_SECTION}'] = ModelCardTemplate.__TRAINING_SECTION__.replace("{LOSS_FUNCTIONS}", info_loss_functions).replace("{FIT_PARAMETERS}", info_fit_parameters)


        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.to(self._target_device)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)


        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = data
                    labels = labels.to(self._target_device)
                    features = list(map(lambda batch: batch_to_device(batch, self._target_device), features))

                    if use_amp:
                        with autocast():
                            loss_value = loss_model(features, labels)

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(features, labels)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)

                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

                if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
                    self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)


            self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

        if evaluator is None and output_path is not None:   #No evaluator, but output path: save final model version
            self.save(output_path)

        if checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)



    def evaluate(self, evaluator: SentenceEvaluator, output_path: str = None):
        """
        Evaluate the model

        :param evaluator:
            the evaluator
        :param output_path:
            the evaluator can write the results to this path
        """
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        return evaluator(self, output_path)

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        """Runs evaluation during the training"""
        eval_path = output_path
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            eval_path = os.path.join(output_path, "eval")
            os.makedirs(eval_path, exist_ok=True)

        if evaluator is not None:
            score = evaluator(self, output_path=eval_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)

    def _save_checkpoint(self, checkpoint_path, checkpoint_save_total_limit, step):
        # Store new checkpoint
        self.save(os.path.join(checkpoint_path, str(step)))

        # Delete old checkpoints
        if checkpoint_save_total_limit is not None and checkpoint_save_total_limit > 0:
            old_checkpoints = []
            for subdir in os.listdir(checkpoint_path):
                if subdir.isdigit():
                    old_checkpoints.append({'step': int(subdir), 'path': os.path.join(checkpoint_path, subdir)})

            if len(old_checkpoints) > checkpoint_save_total_limit:
                old_checkpoints = sorted(old_checkpoints, key=lambda x: x['step'])
                shutil.rmtree(old_checkpoints[0]['path'])


    def _load_auto_model(self, model_name_or_path):
        """
        Creates a simple Transformer + Mean Pooling model and returns the modules
        """
        logger.warning("No sentence-transformers model found with name {}. Creating a new one with MEAN pooling.".format(model_name_or_path))
        transformer_model = Transformer(model_name_or_path)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), 'mean')
        return [transformer_model, pooling_model]

    def _load_sbert_model(self, model_path):
        """
        Loads a full sentence-transformers model
        """
        # Check if the config_sentence_transformers.json file exists (exists since v2 of the framework)
        config_sentence_transformers_json_path = os.path.join(model_path, 'config_sentence_transformers.json')
        if os.path.exists(config_sentence_transformers_json_path):
            with open(config_sentence_transformers_json_path) as fIn:
                self._model_config = json.load(fIn)

            if '__version__' in self._model_config and 'sentence_transformers' in self._model_config['__version__'] and self._model_config['__version__']['sentence_transformers'] > __version__:
                logger.warning("You try to use a model that was created with version {}, however, your version is {}. This might cause unexpected behavior or errors. In that case, try to update to the latest version.\n\n\n".format(self._model_config['__version__']['sentence_transformers'], __version__))

        # Check if a readme exists
        model_card_path = os.path.join(model_path, 'README.md')
        if os.path.exists(model_card_path):
            try:
                with open(model_card_path, encoding='utf8') as fIn:
                    self._model_card_text = fIn.read()
            except:
                pass

        # Load the modules of sentence transformer
        modules_json_path = os.path.join(model_path, 'modules.json')
        with open(modules_json_path) as fIn:
            modules_config = json.load(fIn)

        modules = OrderedDict()
        for module_config in modules_config:
            module_class = import_from_string(module_config['type'])
            module = module_class.load(os.path.join(model_path, module_config['path']))
            modules[module_config['name']] = module

        return modules

    @staticmethod
    def load(input_path):
        return SentenceTransformer(input_path)

    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    @property
    def device(self) -> device:
        """
        Get torch.device from module, assuming that the whole module has one device.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5

            def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].device

    @property
    def tokenizer(self):
        """
        Property to get the tokenizer that is used by this model
        """
        return self._first_module().tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        """
        Property to set the tokenizer that should be used by this model
        """
        self._first_module().tokenizer = value

    @property
    def max_seq_length(self):
        """
        Property to get the maximal input sequence length for the model. Longer inputs will be truncated.
        """
        return self._first_module().max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value):
        """
        Property to set the maximal input sequence length for the model. Longer inputs will be truncated.
        """
        self._first_module().max_seq_length = value
