import json
import logging
import os
import shutil
from collections import OrderedDict
from typing import List, Dict, Tuple, Iterable, Type
from zipfile import ZipFile

import numpy as np
import transformers
import torch
from numpy import ndarray
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from . import __DOWNLOAD_SERVER__
from .evaluation import SentenceEvaluator
from .util import import_from_string, batch_to_device, http_get
from . import __version__

class SentenceTransformer(nn.Sequential):
    def __init__(self, model_name_or_path: str = None, modules: Iterable[nn.Module] = None, device: str = None):
        if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])

        if model_name_or_path is not None and model_name_or_path != "":
            logging.info("Load pretrained SentenceTransformer: {}".format(model_name_or_path))

            if '/' not in model_name_or_path and '\\' not in model_name_or_path and not os.path.isdir(model_name_or_path):
                logging.info("Did not find a '/' or '\\' in the name. Assume to download model from server.")
                model_name_or_path = __DOWNLOAD_SERVER__ + model_name_or_path + '.zip'

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
                    try:
                        zip_save_path = os.path.join(model_path, 'model.zip')
                        http_get(model_url, zip_save_path)
                        with ZipFile(zip_save_path, 'r') as zip:
                            zip.extractall(model_path)
                    except Exception as e:
                        shutil.rmtree(model_path)
                        raise e
            else:
                model_path = model_name_or_path

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


        super().__init__(modules)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info("Use pytorch device: {}".format(device))
        self.device = torch.device(device)
        self.to(device)

    def encode(self, sentences: List[str], batch_size: int = 8, show_progress_bar: bool = None) -> List[ndarray]:
        """
       Computes sentence embeddings

       :param sentences:
           the sentences to embed
       :param batch_size:
           the batch size used for the computation
       :param show_progress_bar:
            Output a progress bar when encode sentences
       :return:
           a list with ndarrays of the embeddings for each sentence
       """
        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel()==logging.INFO or logging.getLogger().getEffectiveLevel()==logging.DEBUG)

        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in sentences])

        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Batches")

        for batch_idx in iterator:
            batch_tokens = []

            batch_start = batch_idx
            batch_end = min(batch_start + batch_size, len(sentences))

            longest_seq = 0

            for idx in length_sorted_idx[batch_start: batch_end]:
                sentence = sentences[idx]
                tokens = self.tokenize(sentence)
                longest_seq = max(longest_seq, len(tokens))
                batch_tokens.append(tokens)

            features = {}
            for text in batch_tokens:
                sentence_features = self.get_sentence_features(text, longest_seq)

                for feature_name in sentence_features:
                    if feature_name not in features:
                        features[feature_name] = []
                    features[feature_name].append(sentence_features[feature_name])

            for feature_name in features:
                features[feature_name] = torch.tensor(np.asarray(features[feature_name])).to(self.device)

            with torch.no_grad():
                embeddings = self.forward(features)
                embeddings = embeddings['sentence_embedding'].to('cpu').numpy()
                all_embeddings.extend(embeddings)

        reverting_order = np.argsort(length_sorted_idx)
        all_embeddings = [all_embeddings[idx] for idx in reverting_order]

        return all_embeddings

    def get_max_seq_length(self):
        if hasattr(self._first_module(), 'max_seq_length'):
            return self._first_module().max_seq_length

        return None

    def tokenize(self, text):
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
                feature_lists[feature_name] = torch.tensor(np.asarray(feature_lists[feature_name]))

            features.append(feature_lists)

        return {'features': features, 'labels': torch.stack(labels)}



    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: SentenceEvaluator,
            epochs: int = 1,
            steps_per_epoch = None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params : Dict[str, object ]= {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            fp16: bool = False,
            fp16_opt_level: str = 'O1',
            local_rank: int = -1
            ):
        """
        Train the model with the given training objective

        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param weight_decay:
        :param scheduler:
        :param warmup_steps:
        :param optimizer:
        :param evaluation_steps:
        :param output_path:
        :param save_best_model:
        :param max_grad_norm:
        :param fp16:
        :param fp16_opt_level:
        :param local_rank:
        :param train_objectives:
            Tuples of DataLoader and LossConfig
        :param evaluator:
        :param epochs:
        :param steps_per_epoch: Train for x steps in each epoch. If set to None, the length of the dataset will be used
        """
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            if os.listdir(output_path):
                raise ValueError("Output directory ({}) already exists and is not empty.".format(
                    output_path))

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        device = self.device

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
            t_total = num_train_steps
            if local_rank != -1:
                t_total = t_total // torch.distributed.get_world_size()

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=t_total)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        if fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            for train_idx in range(len(loss_models)):
                model, optimizer = amp.initialize(loss_models[train_idx], optimizers[train_idx], opt_level=fp16_opt_level)
                loss_models[train_idx] = model
                optimizers[train_idx] = optimizer

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

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

                    features, labels = batch_to_device(data, self.device)
                    loss_value = loss_model(features, labels)


                    if fp16:
                        with amp.scale_loss(loss_value, optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                    else:
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps)
                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

            self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1)

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

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if score > self.best_score and save_best_model:
                self.save(output_path)
                self.best_score = score


    def _get_scheduler(self, optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler
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
