
# Training Overview

```{eval-rst}
.. note::
    The CrossEncoder training approach has not been updated in v3.0 when `training Sentence Transformer models <../sentence_transformer/training_overview.html>`_ was improved. Improving training CrossEncoders is planned for a future major update.
```

The `CrossEncoder` class is a wrapper around Hugging Face `AutoModelForSequenceClassification`, but with some methods to make training and predicting scores a little bit easier. The saved models are 100% compatible with Hugging Face and can also be loaded with their classes.

First, you need some sentence pair data. You can either have a continuous score, like:

```{eval-rst}

.. sidebar:: Documentation

    - :class:`~sentence_transformers.readers.InputExample`
```

```python
from sentence_transformers import InputExample

train_samples = [
    InputExample(texts=["sentence1", "sentence2"], label=0.3),
    InputExample(texts=["Another", "pair"], label=0.8),
]
```

Or you have distinct classes as in the [training_nli.py](../../examples/training/cross-encoder/training_nli.py) example:
```python
from sentence_transformers import InputExample

label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
train_samples = [
    InputExample(texts=["sentence1", "sentence2"], label=label2int["neutral"]),
    InputExample(texts=["Another", "pair"], label=label2int["entailment"]),
]
```

Then, you define the base model and the number of labels. You can take any [Hugging Face pre-trained model](https://huggingface.co/models) that is compatible with AutoModel:
```
model = CrossEncoder('distilroberta-base', num_labels=1)
```

For binary tasks and tasks with continuous scores (like STS), we set num_labels=1. For classification tasks, we set it to the number of labels we have.

```{eval-rst}

We start the training by calling :meth:`CrossEncoder.fit <sentence_transformers.cross_encoder.CrossEncoder.fit>`:

.. sidebar:: Documentation

    - :class:`~sentence_transformers.cross_encoder.CrossEncoder`
    - :meth:`CrossEncoder.fit <sentence_transformers.cross_encoder.CrossEncoder.fit>`

::

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
    )
```