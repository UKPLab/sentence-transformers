from dataclasses import dataclass
from transformers import TrainingArguments as TransformersTrainingArguments


@dataclass
class TrainingArguments(TransformersTrainingArguments):
    round_robin_sampler: bool = False
