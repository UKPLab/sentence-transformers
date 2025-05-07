# Losses

`sentence_transformers.sparse_encoder.losses` defines different loss functions that can be used to fine-tune saprse embedding models on training data. The choice of loss function plays a critical role when fine-tuning the model. It determines how well our embedding model will work for the specific downstream task.

Sadly, there is no "one size fits all" loss function. Which loss function is suitable depends on the available training data and on the target task. Consider checking out the [Loss Overview](../../sparse_encoder/loss_overview.md) to help narrow down your choice of loss function(s).


## SparseMultipleNegativesRankingLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss
```

## SparseMarginMSELoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseMarginMSELoss
```

## SparseDistillKLDivLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseDistillKLDivLoss
``` 

## SpladeLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SpladeLoss
```

## FlopsLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.FlopsLoss
```

## SparseCachedMultipleNegativesRankingLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseCachedMultipleNegativesRankingLoss
```

## SparseTripletLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseTripletLoss
```

## SparseMSELoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseMSELoss
```

## CSRLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.CSRLoss
```
With associated ReconstructionLoss :

```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.CSRReconstructionLoss
```

## SparseCosineSimilarityLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseCosineSimilarityLoss
```

## SparseCoSENTLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseCoSENTLoss
```

## SparseAnglELoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseAnglELoss
```

## SparseGISTEmbedLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseGISTEmbedLoss
```

## SparseCachedGISTEmbedLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseCachedGISTEmbedLoss
```
