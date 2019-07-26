from typing import List

import torch
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import os
from .evaluation import SentenceEvaluator
from .util import batch_to_device
from .models import TransformerModel
from .config import LossFunction
import logging


class TrainConfig:
    """
    Configuration for the training of a Sentence BERT model
    """
    epochs: int
    learning_rate: float
    adam_epsilon: float
    weight_decay: float
    warmup_steps: int
    evaluator: SentenceEvaluator
    evaluation_steps: int
    output_path: str
    save_best_model: bool
    gradient_accumulation_steps: int
    fp16: bool
    fp16_opt_level: str
    local_rank: int
    max_grad_norm: float
    correct_bias:bool

    def __init__(self,
                 epochs: int = 1,
                 learning_rate: float = 2e-5,
                 adam_epsilon: float = 1e-6,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 10000,
                 evaluator: SentenceEvaluator = None,
                 evaluation_steps: int = 0,
                 output_path: str = None,
                 save_best_model: bool = False,
                 gradient_accumulation_steps: int = 1,
                 fp16: bool = False,
                 fp16_opt_level: str = '01',
                 local_rank: int = -1,
                 max_grad_norm: float = 1,
                 correct_bias: bool = False):
        """
        The configuration for the training of a Sentence BERT model
        :param epochs:
            number of epochs for the training
        :param learning_rate:
            the learning rate factor for the schedule
        :param adam_epsilon:
            Epsilon for Adam optimizer.
        :param warmup_steps:
            Linear warmup over warmup_steps
        :param evaluator
            the evaluator used to evaluate the model.
            If this is None, then the model will not be evaluated during training
        :param evaluation_steps:
            each epoch, the model will be evaluated every evaluation_steps steps, in addition to the evaluated
            after each epoch.
            gradient_accumulation_steps is ignored for counting the steps.
            If this is 0, then the model will only be evaluated after each epoch.
            If evaluator is None, then this parameter is ignored
        :param output_path
            the path where evaluation results and models will be saved during training.
            The folder needs to be empty.
            The directories will be created, if they do not exist.
        :param save_best_model
            saves the model with the best evaluation score.
            This requires evaluator to be not None
        :param gradient_accumulation_steps:
            the number of steps during which the gradient is accumulated before the model is updated
        :param fp16:
            train with float16 precision and distributed using apex (https://www.github.com/nvidia/apex)
        :param fp16_opt_level:
            For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
            See details at https://nvidia.github.io/apex/amp.html"
        :param local_rank:
            the local rank when using distributed training
        :param correct_bias
            Set to false, to reproduce BertAdam specific behavior
        """
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.evaluator = evaluator
        self.evaluation_steps = evaluation_steps
        self.output_path = output_path
        self.save_best_model = save_best_model
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16
        self.fp16_opt_level = fp16_opt_level
        self.local_rank = local_rank
        self.max_grad_norm = max_grad_norm
        self.correct_bias = correct_bias


class SentenceTrainer:
    """
    Wrapper for the training of a Sentence BERT model
    """
    def __init__(self, sentence_bert: TransformerModel):
        """
        Creates a new trainer for the given model

        :param sentence_bert:
            the model that will be trained
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = sentence_bert

    def save(self, path: str, save_config: bool = True, save_model: bool = True):
        """
        Save the model at the given path

        Directories are created if they do not exist yet.

        :param path:
            path where the model will be saved
        """
        os.makedirs(path, exist_ok=True)
        logging.info("Save the model to " + path)


        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(path, WEIGHTS_NAME)
        output_model_config_file = os.path.join(path, CONFIG_NAME)
        output_sentence_transformer_config_file = os.path.join(path, 'sentence_transformer_config.json')

        if save_config:
            self.model.model_config.to_json_file(output_model_config_file)
            self.model.sentence_transformer_config.to_json_file(output_sentence_transformer_config_file)

        if save_model:
            torch.save(self.model.state_dict(), output_model_file)


    def train(self, dataloader: DataLoader, train_config: TrainConfig):
        """
        Train the model with the given data and config

        :param dataloader:
            the data for the training
        :param train_config:
            the configuration for the training
        """
        if train_config.output_path is not None:
            os.makedirs(train_config.output_path, exist_ok=True)
            if os.listdir(train_config.output_path):
                raise ValueError("Output directory ({}) already exists and is not empty.".format(
                    train_config.output_path))

            self.save(train_config.output_path, save_config=True, save_model=False)

        self.best_score = -9999
        num_train_steps = int(len(dataloader) / train_config.gradient_accumulation_steps * train_config.epochs)

        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': train_config.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = num_train_steps
        if train_config.local_rank != -1:
            t_total = t_total // torch.distributed.get_world_size()

        optimizer = AdamW(optimizer_grouped_parameters, lr=train_config.learning_rate,
                          eps=train_config.adam_epsilon, correct_bias=train_config.correct_bias)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_config.warmup_steps, t_total=t_total)

        if train_config.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(self.model, optimizer, opt_level=train_config.fp16_opt_level)


        global_step = 0

        for epoch in trange(train_config.epochs, desc="Epoch"):
            training_steps = 0
            self.model.train()
            for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
                batch = batch_to_device(batch, self.device)
                input_ids, segment_ids, input_masks, label_ids = batch
                loss = self.model(input_ids, segment_ids, input_masks, label_ids)

                if train_config.gradient_accumulation_steps > 1:
                    loss = loss / train_config.gradient_accumulation_steps

                if train_config.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), train_config.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_config.max_grad_norm)

                training_steps += 1
                if (step + 1) % train_config.gradient_accumulation_steps == 0:
                    scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if train_config.evaluation_steps > 0 and training_steps % train_config.evaluation_steps == 0:
                    self._eval_during_training(train_config, epoch, training_steps)
                    self.model.train()

            self._eval_during_training(train_config, epoch, -1)

    def multitask_train(self, dataloaders: List[DataLoader], losses: List[LossFunction], train_config: TrainConfig):
        """
        Train the model with the given data and config with the given loss for each dataset

        Each dataloader is sampled in turn for one batch.
        We sample only as many batches from each dataloader as there are in the smallest one
        to make sure of equal training with each dataset.

        :param dataloaders:
            the data for the training
        :param losses:
            the losses for the dataloaders
            the losses still uses the configuration as given in sbert_config, so you cannot for example
            have two different SBERTLossFunction.SOFTMAX with different number of labels
        :param train_config:
            the configuration for the training
        """
        if train_config.output_path is not None:
            os.makedirs(train_config.output_path, exist_ok=True)
            if os.listdir(train_config.output_path):
                raise ValueError("Output directory ({}) already exists and is not empty.".format(
                    train_config.output_path))

            self.save(train_config.output_path, save_config=True, save_model=False)

        self.best_score = -9999

        min_batches = min([len(dataloader) for dataloader in dataloaders])
        num_dataloaders = len(dataloaders)
        num_train_steps = int(num_dataloaders*min_batches / train_config.gradient_accumulation_steps * train_config.epochs)

        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = num_train_steps
        if train_config.local_rank != -1:
            t_total = t_total // torch.distributed.get_world_size()

        optimizer = AdamW(optimizer_grouped_parameters, lr=train_config.learning_rate,
                          eps=train_config.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_config.warmup_steps, t_total=t_total)

        if train_config.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(self.model, optimizer, opt_level=train_config.fp16_opt_level)

        global_step = 0

        for epoch in trange(train_config.epochs, desc="Epoch"):
            training_steps = 0
            self.model.train()
            iterators = [iter(dataloader) for dataloader in dataloaders]
            for step in trange(num_dataloaders*min_batches, desc="Iteration"):
                idx = step % num_dataloaders
                batch = batch_to_device(next(iterators[idx]), self.device)
                input_ids, segment_ids, input_masks, label_ids = batch
                loss = self.model(input_ids, segment_ids, input_masks, label_ids, losses[idx])

                if train_config.gradient_accumulation_steps > 1:
                    loss = loss / train_config.gradient_accumulation_steps

                if train_config.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), train_config.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_config.max_grad_norm)

                training_steps += 1
                if (step + 1) % train_config.gradient_accumulation_steps == 0:
                    scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if train_config.evaluation_steps > 0 and training_steps % train_config.evaluation_steps == 0:
                    self._eval_during_training(train_config, epoch, training_steps)
                    self.model.train()

            self._eval_during_training(train_config, epoch, -1)

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
        evaluator(self.model, output_path)

    def _eval_during_training(self, train_config, epoch, steps):
        if train_config.evaluator is not None:
            score = train_config.evaluator(self.model, output_path=train_config.output_path, epoch=epoch, steps=steps)
            if score > self.best_score and train_config.save_best_model:
                self.save(train_config.output_path, save_model=True, save_config=True)
                self.best_score = score

    @staticmethod
    def warmup_linear(x: int, warmup: float = 0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x
