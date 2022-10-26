""" A Trainer that is compatible with Huggingface transformers """
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase, Trainer
from transformers.tokenization_utils import BatchEncoding
from transformers.utils.generic import PaddingStrategy

from sentence_transformers import SentenceTransformer


@dataclass
class SentenceTransformersCollator:
    """Collator for a SentenceTransformers model.
    This encodes the text columns to {column}_input_ids and {column}_attention_mask columns.
    This works with the two text dataset that is used as the example in the training overview:
    https://www.sbert.net/docs/training/overview.html"""

    tokenizer: PreTrainedTokenizerBase
    text_columns: List[str]

    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __init__(self, tokenizer: AutoTokenizer, text_columns: List[str]) -> None:
        self.tokenizer = tokenizer
        self.text_columns = text_columns

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {"label": torch.tensor([row["label"] for row in features])}
        for column in self.text_columns:
            padded = self._encode([row[column] for row in features])
            batch[f"{column}_input_ids"] = padded.input_ids
            batch[f"{column}_attention_mask"] = padded.attention_mask
        return batch

    def _encode(self, texts: List[str]) -> BatchEncoding:
        tokens = self.tokenizer(texts, return_attention_mask=False)
        return self.tokenizer.pad(
            tokens,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )


class SentenceTransformersTrainer(Trainer):
    """Huggingface Trainer for a SentenceTransformers model.

    This works with the two text dataset that is used as the example in the training overview:
    https://www.sbert.net/docs/training/overview.html

    You use this by providing the loss function and the sentence transformer model.
    An example that replicates the quickstart is:

    >> from sentence_transformers import SentenceTransformer, losses, evaluation
    >> import datasets # huggingface library that is separate to transformers
    >> from transformers import TrainingArguments, EvalPrediction

    >> sick_ds = datasets.load_dataset("sick")

    >> text_columns = ["sentence_A", "sentence_B"]
    >> model = SentenceTransformer("distilbert-base-nli-mean-tokens")
    >> tokenizer = model.tokenizer
    >> loss = losses.CosineSimilarityLoss(model)
    >> data_collator = SentenceTransformersCollator(tokenizer=tokenizer)

    >> evaluator = evaluation.EmbeddingSimilarityEvaluator(
    >>     sick_ds["validation"]["sentence_A"],
    >>     sick_ds["validation"]["sentence_B"],
    >>     sick_ds["validation"]["label"],
    >>     main_similarity=evaluation.SimilarityFunction.COSINE,
    >> )
    >> def compute_metrics(predictions: EvalPrediction) -> Dict[str, float]:
    >>     return {
    >>         "cosine_similarity": evaluator(model)
    >>     }

    >> training_arguments = TrainingArguments(
    >>     report_to="none",
    >>     output_dir=run_folder,
    >>     num_train_epochs=10,
    >>     seed=33,
    >>     # checkpoint settings
    >>     logging_dir=run_folder / "logs",
    >>     save_total_limit=2,
    >>     load_best_model_at_end=True,
    >>     metric_for_best_model="cosine_similarity",
    >>     greater_is_better=True,
    >>     # needed to get sentence_A and sentence_B
    >>     remove_unused_columns=False,
    >> )

    >> trainer = SentenceTransformersTrainer(
    >>     model=model,
    >>     args=training_arguments,
    >>     train_dataset=sick_ds["train"],
    >>     eval_dataset=sick_ds["validation"],
    >>     data_collator=data_collator,
    >>     tokenizer=tokenizer,
    >>     loss=loss,
    >>     compute_metrics=compute_metrics,
    >> )
    >> trainer.train()
    """

    def __init__(
        self,
        *args,
        text_columns: List[str],
        loss: nn.Module,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.text_columns = text_columns
        self.loss = loss
        self.loss.to(self.model.device)

    def compute_loss(
        self,
        model: SentenceTransformer,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        pad_token_id = model.tokenizer.pad_token_id
        features = [
            {
                "input_ids": input_ids,
                "attention_mask": (input_ids != pad_token_id).to(int),
            }
            for input_ids in [inputs["sentence_A"], inputs["sentence_B"]]
        ]
        loss = self.loss(features, inputs["label"])
        if return_outputs:
            output = torch.cat(
                [model(row)["sentence_embedding"][:, None] for row in features], dim=1
            )
            return loss, output
        return loss

    def collect_features(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Turn the inputs from the dataloader into the separate model inputs."""
        return [
            {
                "input_ids": inputs[f"{column}_input_ids"],
                "attention_mask": inputs[f"{column}_attention_mask"],
            }
            for column in self.text_columns
        ]
