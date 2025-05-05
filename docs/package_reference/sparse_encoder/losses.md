# Losses

`sentence_transformers.sparse_encoder.losses` defines different loss functions that can be used to fine-tune saprse embedding models on training data. The choice of loss function plays a critical role when fine-tuning the model. It determines how well our embedding model will work for the specific downstream task.

Sadly, there is no "one size fits all" loss function. Which loss function is suitable depends on the available training data and on the target task. Consider checking out the [Loss Overview](../../sparse_encoder/loss_overview.md) to help narrow down your choice of loss function(s).


## SparseMultipleNegativesRankingLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss
   :members:
```

## SparseCachedMultipleNegativesRankingLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseCachedMultipleNegativesRankingLoss
   :members:
```

## SparseTripletLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseTripletLoss
   :members:
```

## SparseMSELoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseMSELoss
   :members:
```

## SparseMarginMSELoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseMarginMSELoss
   :members:
```

## SpladeLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SpladeLoss
   :members:
```

## CSRLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.CSRLoss
   :members:
```

## CSRReconstructionLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.CSRReconstructionLoss
   :members:
```

## FlopsLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.FlopsLoss
   :members:
```

## SparseCosineSimilarityLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseCosineSimilarityLoss
   :members:
```

## SparseCoSENTLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseCoSENTLoss
   :members:
```

## SparseAnglELoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseAnglELoss
   :members:
```

## SparseGISTEmbedLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseGISTEmbedLoss
   :members:
```

## SparseCachedGISTEmbedLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseCachedGISTEmbedLoss
   :members:
```

## SparseDistillKLDivLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseDistillKLDivLoss
   :members:
``` 