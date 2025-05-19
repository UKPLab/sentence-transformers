# Modules

`sentence_transformers.sparse_encoder.models` defines different building blocks, that can be used to create SparseEncoder networks from scratch. For more details, see [Training Overview](../../sparse_encoder/training_overview.md).
Note that modules from `sentence_transformers.models` can also be used for Sparse models, such as `sentence_transformers.models.Transformer` from [SentenceTransformer > Modules](../sentence_transformer/models.md)
## SPLADE Pooling
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.models.SpladePooling
```

## MLM Transformer
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.models.MLMTransformer
```

## CSR Sparsity
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.models.CSRSparsity
```

## IDF
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.models.IDF
``` 