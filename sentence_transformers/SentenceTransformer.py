import json
import logging
import os
import shutil
from collections import OrderedDict
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable
from zipfile import ZipFile
import requests
import numpy as np
import transformers
import torch
from numpy import ndarray
from torch import nn, Tensor, device
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
import torch.multiprocessing as mp
import math
import queue

from . import __DOWNLOAD_SERVER__
from .evaluation import SentenceEvaluator
from .util import import_from_string, batch_to_device, http_get
from .datasets.EncodeDataset import EncodeDataset
from .models import Transformer, Pooling
from . import __version__

class SentenceTransformer(nn.Sequential):
    def __init__(self, model_name_or_path: str = None, modules: Iterable[nn.Module] = None, device: str = None):
        if model_name_or_path is not None and model_name_or_path != "":
            logging.info("Load pretrained SentenceTransformer: {}".format(model_name_or_path))
            model_path = model_name_or_path

            if not os.path.isdir(model_path) and not model_path.startswith('http://') and not model_path.startswith('https://'):
                logging.info("Did not find folder {}. Assume to download model from server.".format(model_path))
                model_path = __DOWNLOAD_SERVER__ + model_path + '.zip'

            if model_path.startswith('http://') or model_path.startswith('https://'):
                model_url = model_path
                folder_name = model_url.replace("https://", "").replace("http://", "").replace("/", "_")[:250].rstrip('.zip')

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
                    if model_url[-1] == "/":
                        model_url = model_url[:-1]
                    logging.info("Downloading sentence transformer model from {} and saving it at {}".format(model_url, model_path))
                    try:
                        zip_save_path = os.path.join(model_path, 'model.zip')
                        http_get(model_url, zip_save_path)
                        with ZipFile(zip_save_path, 'r') as zip:
                            zip.extractall(model_path)
                        os.remove(zip_save_path)
                    except requests.exceptions.HTTPError as e:
                        shutil.rmtree(model_path)
                        if e.response.status_code == 404:
                            logging.warning('SentenceTransformer-Model {} not found. Try to create it from scratch'.format(model_url))
                            logging.warning('Try to create Transformer Model {} with mean pooling'.format(model_name_or_path))

                            model_path = None
                            transformer_model = Transformer(model_name_or_path)
                            pooling_model = Pooling(transformer_model.get_word_embedding_dimension())
                            modules = [transformer_model, pooling_model]

                        else:
                            raise e
                    except Exception as e:
                        shutil.rmtree(model_path)
                        raise e




            #### Load from disk
            if model_path is not None:
                logging.info("Load SentenceTransformer from folder: {}".format(model_path))

                if os.path.exists(os.path.join(model_path, 'config.json')):
                    with open(os.path.join(model_path, 'config.json')) as fIn:
                        config = json.load(fIn)
                        if config['__version__'] > __version__:
                            logging.warning("You try to use a model that was created with version {}, however, your version is {}. This might cause unexpected behavior or errors. In that case, try to update to the latest version.\n\n\n".format(config['__version__'], __version__))

                with open(os.path.join(model_path, 'modules.json')) as fIn:
                    contained_modules = json.load(fIn)

                modules = OrderedDict()
                for module_config in contained_modules:
                    module_class = import_from_string(module_config['type'])
                    module = module_class.load(os.path.join(model_path, module_config['path']))
                    modules[module_config['name']] = module


        if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])

        super().__init__(modules)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)


    def encode(self, sentences: Union[str, List[str], List[int]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               is_pretokenized: bool = False,
               device: str = None,
               num_workers: int = 0) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings


        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings.
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from conver_to_numpy
        :param is_pretokenized: If is_pretokenized=True, sentences must be a list of integers, containing the tokenized sentences with each token convert to the respective int.
        :param device: Which torch.device to use for the computation
        :param num_workers: Number of background-workers to tokenize data. Set to positive number to increase tokenization speed
        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel()==logging.INFO or logging.getLogger().getEffectiveLevel()==logging.DEBUG)

        input_was_string = False
        if isinstance(sentences, str): #Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = self._target_device

        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        inp_dataset = EncodeDataset(sentences_sorted, model=self, is_tokenized=is_pretokenized)
        inp_dataloader = DataLoader(inp_dataset, batch_size=batch_size, collate_fn=self.smart_batching_collate_text_only, num_workers=num_workers, shuffle=False)

        iterator = inp_dataloader
        if show_progress_bar:
            iterator = tqdm(inp_dataloader, desc="Batches")

        for features in iterator:
            for feature_name in features:
                features[feature_name] = features[feature_name].to(device)

            with torch.no_grad():
                out_features = self.forward(features)
                embeddings = out_features[output_value]

                if output_value == 'token_embeddings':
                    #Set token embeddings to 0 for padding tokens
                    input_mask = out_features['attention_mask']
                    input_mask_expanded = input_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    embeddings = embeddings * input_mask_expanded

                all_embeddings.extend(embeddings)


        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.cpu().detach().numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings



    def start_multi_process_pool(self, target_devices: List[str] = None, encode_batch_size: int = 32):
        """
        Starts multi process to process the encode with several, independent  process.
        This methos is recommend if you want to encode on multiple GPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process

        :param target_devices: PyTorch target devices, e.g. cuda:0, cuda:1... If None, all available CUDA devices will be used
        :param encode_batch_size: Batch size for each process when calling encode
        :return: Returns a dict with the target processes, an input queue and and output queue.
        """
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
            else:
                logging.info("CUDA is not available. Start 4 CPU worker")
                target_devices = ['cpu']*4

        logging.info("Start multi-process pool on devices: {}".format(', '.join(map(str, target_devices))))

        ctx = mp.get_context('spawn')
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for cuda_id in target_devices:
            p = ctx.Process(target=SentenceTransformer._encode_multi_process_worker, args=(cuda_id, self, input_queue, output_queue, encode_batch_size), daemon=True)
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


    def encode_multi_process(self, sentences: List[str], pool: Dict[str, object], is_pretokenized: bool = False):
        """
        This method allows to run encode() on multiple GPUs. The sentences are chunked into smaller packages
        and sent to individual processes, which encode these on the different GPUs. This method is only suitable
        for encoding large sets of sentences

        :param sentences: List of sentences
        :param pool: A pool of workers started with SentenceTransformer.start_multi_process_pool
        :param is_pretokenized: If true, no tokenization will be applied. It is expected that the input sentences are list of ints.
        :return: Numpy matrix with all embeddings
        """

        chunk_size = min(math.ceil(len(sentences) / len(pool["processes"]) / 10), 5000)
        logging.info("Chunk data into packages of size {}".format(chunk_size))

        if is_pretokenized:
            sentences_tokenized = sentences
        else:
            sentences_tokenized = map(self.tokenize, sentences)

        input_queue = pool['input']
        num_chunks = 0
        chunk = []

        for sentence in sentences_tokenized:
            chunk.append(sentence)
            if len(chunk) >= chunk_size:
                input_queue.put([num_chunks, chunk])
                num_chunks += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([num_chunks, chunk])
            num_chunks += 1

        output_queue = pool['output']
        results_list = sorted([output_queue.get() for _ in range(num_chunks)], key=lambda x: x[0])
        embeddings = np.concatenate([result[1] for result in results_list])
        return embeddings

    @staticmethod
    def _encode_multi_process_worker(target_device: str, model, input_queue, results_queue, encode_batch_size):
        """
        Internal working process to encode sentences in multi-process setup
        """
        while True:
            try:
                id, sentences = input_queue.get()
                embeddings = model.encode(sentences, device=target_device, is_pretokenized=True, show_progress_bar=False, convert_to_numpy=True, batch_size=encode_batch_size)
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

    def tokenize(self, text: str):
        """
        Tokenizes the text
        """
        return self._first_module().tokenize(text)

    def get_sentence_features(self, *features):
        return self._first_module().get_sentence_features(*features)

    def get_sentence_embedding_dimension(self):
        return self._last_module().get_sentence_embedding_dimension()

    def _first_module(self):
        """Returns the first module of this sequential embedder"""
        return self._modules[next(iter(self._modules))]

    def _last_module(self):
        """Returns the last module of this sequential embedder"""
        return self._modules[next(reversed(self._modules))]

    def save(self, path):
        """
        Saves all elements for this seq. sentence embedder into different sub-folders
        """
        if path is None:
            return

        logging.info("Save model to {}".format(path))
        contained_modules = []

        for idx, name in enumerate(self._modules):
            module = self._modules[name]
            model_path = os.path.join(path, str(idx)+"_"+type(module).__name__)
            os.makedirs(model_path, exist_ok=True)
            module.save(model_path)
            contained_modules.append({'idx': idx, 'name': name, 'path': os.path.basename(model_path), 'type': type(module).__module__})

        with open(os.path.join(path, 'modules.json'), 'w') as fOut:
            json.dump(contained_modules, fOut, indent=2)

        with open(os.path.join(path, 'config.json'), 'w') as fOut:
            json.dump({'__version__': __version__}, fOut, indent=2)

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0][0])

        labels = []
        paired_texts = [[] for _ in range(num_texts)]
        max_seq_len = [0] * num_texts
        for tokens, label in batch:
            labels.append(label)
            for i in range(num_texts):
                paired_texts[i].append(tokens[i])
                max_seq_len[i] = max(max_seq_len[i], len(tokens[i]))

        features = []
        for idx in range(num_texts):
            max_len = max_seq_len[idx]
            feature_lists = {}

            for text in paired_texts[idx]:
                sentence_features = self.get_sentence_features(text, max_len)

                for feature_name in sentence_features:
                    if feature_name not in feature_lists:
                        feature_lists[feature_name] = []

                    feature_lists[feature_name].append(sentence_features[feature_name])


            for feature_name in feature_lists:
                #feature_lists[feature_name] = torch.tensor(np.asarray(feature_lists[feature_name]))
                feature_lists[feature_name] = torch.cat(feature_lists[feature_name])

            features.append(feature_lists)

        return {'features': features, 'labels': torch.stack(labels)}


    def smart_batching_collate_text_only(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """

        max_seq_len = max([len(text) for text in batch])
        feature_lists = {}

        for text in batch:
            sentence_features = self.get_sentence_features(text, max_seq_len)
            for feature_name in sentence_features:
                if feature_name not in feature_lists:
                    feature_lists[feature_name] = []

                feature_lists[feature_name].append(sentence_features[feature_name])

        for feature_name in feature_lists:
            feature_lists[feature_name] = torch.cat(feature_lists[feature_name])

        return feature_lists



    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: SentenceEvaluator,
            epochs: int = 1,
            steps_per_epoch = None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params : Dict[str, object]= {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            output_path_ignore_not_empty: bool = False,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
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
        :param output_path_ignore_not_empty: By default, training will stop if output_path is not empty. If set to true, this error will be ignored and training proceeds.
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        """

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            if not output_path_ignore_not_empty and len(os.listdir(output_path)) > 0:
                raise ValueError("Output directory ({}) already exists and is not empty.".format(
                    output_path))

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        device = self._target_device

        for loss_model in loss_models:
            loss_model.to(device)

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
        for epoch in trange(epochs, desc="Epoch"):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        #logging.info("Restart data_iterator")
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = batch_to_device(data, self._target_device)

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
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch,
                                               training_steps, callback)
                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

            self._eval_during_training(evaluator, output_path, save_best_model, epoch,
                                       -1, callback)

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
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score and save_best_model:
                self.save(output_path)
                self.best_score = score


    def _get_scheduler(self, optimizer, scheduler: str, warmup_steps: int, t_total: int):
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
        Property to set the tokenizer that is should used by this model
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
