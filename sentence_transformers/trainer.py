import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from transformers import PreTrainedTokenizerBase, Trainer, EvalPrediction, TrainerCallback
from transformers.trainer import TRAINING_ARGS_NAME

from datasets import DatasetDict, Dataset
from transformers.trainer_utils import EvalLoopOutput
from transformers.data.data_collator import DataCollator

from sentence_transformers.training_args import TrainingArguments
from sentence_transformers import SentenceTransformer
from sentence_transformers.data_collator import SentenceTransformerDataCollator
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.sampler import RoundRobinBatchSampler

logger = logging.getLogger(__name__)


class SentenceTransformerTrainer(Trainer):
    def __init__(
        self,
        model: Optional[SentenceTransformer] = None,
        args: TrainingArguments = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        loss: nn.Module = None,
        evaluator: SentenceEvaluator = None,
        data_collator: Optional[DataCollator] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], SentenceTransformer]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        if tokenizer is None:
            tokenizer = model.tokenizer
        if data_collator is None:
            data_collator = SentenceTransformerDataCollator(tokenizer=tokenizer)
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.loss = loss
        if isinstance(loss, dict):
            self.loss = {dataset_name: loss_fn.to(self.model.device) for dataset_name, loss_fn in loss.items()}
            # TODO: Additional conditionals to check whether the dict points to dataset names correctly
        else:
            self.loss.to(self.model.device)
        self.evaluator = evaluator
        self.training_with_dataset_dict = isinstance(self.train_dataset, DatasetDict)
        if self.training_with_dataset_dict:
            self.dataset_names = list(self.train_dataset.keys())
            self.dataset_idx = 0

    def compute_loss(
        self,
        model: SentenceTransformer,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        features = self.collect_features(inputs)
        loss_fn = self.loss

        if self.training_with_dataset_dict and isinstance(loss_fn, dict):
            loss_fn = self.loss[self.dataset_name]

        loss = loss_fn(features, inputs["label"] if "label" in inputs else None)
        if return_outputs:
            output = torch.cat([model(row)["sentence_embedding"][:, None] for row in features], dim=1)
            return loss, output
        return loss

    def collect_features(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> List[Dict[str, torch.Tensor]]:
        """Turn the inputs from the dataloader into the separate model inputs."""
        features = []
        for column in inputs:
            if column.endswith("_input_ids"):
                features.append(
                    {
                        "input_ids": inputs[column],
                        "attention_mask": inputs[column.replace("_input_ids", "_attention_mask")],
                    }
                )
        return features

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        if (
            self.training_with_dataset_dict
            and self.is_in_train
            and isinstance(self.eval_dataset, dict)
            and metric_key_prefix.startswith("eval_")
            and metric_key_prefix[5:] in list(self.eval_dataset.keys())
        ):
            # TODO: What if the evaluation dataset_dict has different keys than the loss one?
            self.dataset_name = metric_key_prefix[5:]

        output = super().evaluation_loop(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # If the evaluator is not defined, we can just return the output
        if self.evaluator is None:
            return output

        # If we are training and eval_dataset is a DatasetDict, then we should
        # 1) only run the evaluator for the first dataset
        # 2) prefix that only run as "eval", rather than e.g. "eval_multi_nli"
        if self.is_in_train and isinstance(self.eval_dataset, dict) and metric_key_prefix.startswith("eval_"):
            if metric_key_prefix[5:] == list(self.eval_dataset.keys())[0]:
                metric_key_prefix = "eval"
            else:
                return output

        evaluator_metrics = self.evaluator(self.model)
        if not isinstance(evaluator_metrics, dict):
            evaluator_metrics = {"evaluator": evaluator_metrics}

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(evaluator_metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                evaluator_metrics[f"{metric_key_prefix}_{key}"] = evaluator_metrics.pop(key)

        output.metrics.update(evaluator_metrics)

        return output

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        if isinstance(train_dataset, DatasetDict):
            sizes = [len(ds) for ds in train_dataset.values()]
            # TODO: Maybe I can infer from the sampler index which dataset it is?
            # for dataset_name, dataset in train_dataset.items():
            #     train_dataset[dataset_name] = dataset.add_column("dataset_name", [dataset_name] * len(dataset))
            train_dataset = ConcatDataset(train_dataset.values())
        else:
            sizes = [len(train_dataset)]

        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "batch_sampler": RoundRobinBatchSampler(
                lengths=sizes,
                batch_size=self.args.train_batch_size,
                drop_last=self.args.dataloader_drop_last,
                seed=self.args.seed,
                trainer=self,
            ),
        }

        self._train_dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        return self._train_dataloader

    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            # Prevent errors if the evaluator is set but no eval_dataset is provided
            if self.evaluator is not None:
                return DataLoader([])
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        if isinstance(eval_dataset, DatasetDict):
            sizes = [len(ds) for ds in eval_dataset.values()]
            eval_dataset = ConcatDataset(eval_dataset.values())
        else:
            sizes = [len(eval_dataset)]

        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "batch_sampler": RoundRobinBatchSampler(
                lengths=sizes,
                batch_size=self.args.train_batch_size,
                drop_last=self.args.dataloader_drop_last,
                seed=self.args.seed,
            ),
        }

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

    # TODO: Also override the test_dataloader

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save(output_dir, safe_serialization=self.args.save_safetensors)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
