# Losses
`sentence_transformers.cross_encoder.losses` defines different loss functions that can be used to fine-tune cross-encoder models on training data. The choice of loss function plays a critical role when fine-tuning the model. It determines how well our model will work for the specific downstream task.

Sadly, there is no "one size fits all" loss function. Which loss function is suitable depends on the available training data and on the target task. Consider checking out the [Loss Overview](../../cross_encoder/loss_overview.md) to help narrow down your choice of loss function(s).

## BinaryCrossEntropyLoss
```{eval-rst}
.. autoclass:: sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss
```

## CrossEntropyLoss
```{eval-rst}
.. autoclass:: sentence_transformers.cross_encoder.losses.CrossEntropyLoss
```

## MultipleNegativesRankingLoss
```{eval-rst}
.. autoclass:: sentence_transformers.cross_encoder.losses.MultipleNegativesRankingLoss
```

## CachedMultipleNegativesRankingLoss
```{eval-rst}
.. autoclass:: sentence_transformers.cross_encoder.losses.CachedMultipleNegativesRankingLoss
```

## MSELoss
```{eval-rst}
.. autoclass:: sentence_transformers.cross_encoder.losses.MSELoss
```
