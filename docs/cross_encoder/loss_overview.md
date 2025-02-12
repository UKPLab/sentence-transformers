# Loss Overview

Loss functions play a critical role in the performance of your fine-tuned Cross Encoder model. Sadly, there is no "one size fits all" loss function. Ideally, this table should help narrow down your choice of loss function(s) by matching them to your data formats.

```{eval-rst}
.. note:: 

    You can often convert one training data format into another, allowing more loss functions to be viable for your scenario. For example, ``(sentence_A, sentence_B) pairs`` with ``class`` labels can be converted into ``(anchor, positive, negative) triplets`` by sampling sentences with the same or different classes.
```

| Inputs                                            | Labels                         | Appropriate Loss Functions                                                                                                                                                                                                                                              |
|---------------------------------------------------|--------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `(sentence_A, sentence_B) pairs`                  | `class`                        | <a href="../package_reference/cross_encoder/losses.html#crossentropyloss">`CrossEntropyLoss`</a>                                                                                                                                                                        |
| `(anchor, positive) pairs`                        | `none`                         | <a href="../package_reference/cross_encoder/losses.html#multiplenegativesrankingloss">`MultipleNegativesRankingLoss`</a><br><a href="../package_reference/sentence_transformer/losses.html#cachedmultiplenegativesrankingloss">`CachedMultipleNegativesRankingLoss`</a> |
| `(anchor, positive/negative) pairs`               | `1 if positive, 0 if negative` | <a href="../package_reference/cross_encoder/losses.html#binarycrossentropyloss">`BinaryCrossEntropyLoss`</a>                                                                                                                                                            |
| `(sentence_A, sentence_B) pairs`                  | `float similarity score`       | <a href="../package_reference/cross_encoder/losses.html#binarycrossentropyloss">`BinaryCrossEntropyLoss`</a>                                                                                                                                                            |
| `(anchor, positive, negative) triplets`           | `none`                         | <a href="../package_reference/cross_encoder/losses.html#multiplenegativesrankingloss">`MultipleNegativesRankingLoss`</a><br><a href="../package_reference/sentence_transformer/losses.html#cachedmultiplenegativesrankingloss">`CachedMultipleNegativesRankingLoss`</a> |
| `(anchor, positive, negative_1, ..., negative_n)` | `none`                         | <a href="../package_reference/cross_encoder/losses.html#multiplenegativesrankingloss">`MultipleNegativesRankingLoss`</a><br><a href="../package_reference/sentence_transformer/losses.html#cachedmultiplenegativesrankingloss">`CachedMultipleNegativesRankingLoss`</a> |

## Distillation
These loss functions are specifically designed to be used when distilling the knowledge from one model into another.
For example, when finetuning a small model to behave more like a larger & stronger one, or when finetuning a model to become multi-lingual.

| Texts                            | Labels             | Appropriate Loss Functions                                                     |
|----------------------------------|--------------------|--------------------------------------------------------------------------------|
| `(sentence_A, sentence_B) pairs` | `similarity score` | <a href="../package_reference/cross_encoder/losses.html#mseloss">`MSELoss`</a> |

## Commonly used Loss Functions
In practice, not all loss functions get used equally often. The most common scenarios are:

* TODO

## Custom Loss Functions

```{eval-rst}
Advanced users can create and train with their own loss functions. Custom loss functions only have a few requirements:

- They must be a subclass of :class:`torch.nn.Module`.
- They must have ``model`` as the first argument in the constructor.
- They must implement a ``forward`` method that accepts ``inputs`` and ``labels``. The former is a nested list of texts in the batch, with each element in the outer list representing a column in the training dataset. You have to combine these texts into pairs that can be 1) tokenized and 2) fed to the model. The latter is an optional tensor of labels from a ``label`` or ``score`` column in the dataset. The method must return a single loss value.

To get full support with the automatic model card generation, you may also wish to implement:

- a ``get_config_dict`` method that returns a dictionary of loss parameters.
- a ``citation`` property so your work gets cited in all models that train with the loss.
```