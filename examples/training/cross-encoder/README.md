# Cross-Encoders
SentenceTransformers also supports the option to train Cross-Encoder for sentence pair score and sentence pair classification tasks. For the what Cross-Encoders are and the difference between Cross- and Bi-Encoders, see [Cross-Encoders](../../applications/cross-encoder/README.md).

## Examples
See the following examples how to train Cross-Encoders:
- [training_stsbenchmark.py](training_stsbenchmark.py) - Example how to train for Semantic Textual Similarity (STS) on the STS benchmark dataset.
- [training_quora_duplicate_questions.py](training_quora_duplicate_questions.py) - Example how to train a Cross-Encoder to predict if two questions are duplicates. Uses Quora Duplicate Questions as training dataset.
- [training_nli.py](training_nli.py) - Example for a multilabel classification task for Natural Language Inference (NLI) task.

## Training CrossEncoders

The `CrossEncoder` class is a wrapper around Huggingface `AutoModelForSequenceClassification`, but with some methods to make training and predicting scores a little bit easier. The saved models are 100% compatible with Huggingface and can also be loaded with their classes.

First, you need some sentence pair data. You can either have a continious score, like:
```python
from sentence_transformers import InputExample
train_samples = [
  InputExample(texts=['sentence1', 'sentence2'], label=0.3),
  InputExample(texts=['Another', 'pair'], label=0.8),
]
```

Or you have distinct classes as in the [training_nli.py](training_nli.py) example:
```python
from sentence_transformers import InputExample
label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
train_samples = [
  InputExample(texts=['sentence1', 'sentence2'], label=label2int['neutral']),
  InputExample(texts=['Another', 'pair'], label=label2int['entailment']),
]
```

Then, you define the base model and the number of labels. You can take any [Huggingface pre-trained model](https://huggingface.co/transformers/pretrained_models.html) that is compatible with AutoModel:
```
model = CrossEncoder('distilroberta-base', num_labels=1)
```

For binary tasks and tasks with continious scores (like STS), we set num_labels=1. For classification tasks, we set it to the number of labels we have.

We start the training by calling `model.fit()`:
```python
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=model_save_path)
```



