from dataclasses import dataclass
from transformers import TrainingArguments as TransformersTrainingArguments


# TODO: Improve the documentation here
@dataclass
class SentenceTransformerTrainingArguments(TransformersTrainingArguments):
    round_robin_sampler: bool = False
