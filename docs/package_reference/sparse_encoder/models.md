# Models

`sentence_transformers.sparse_encoder.models` defines different building blocks, that can be used to create SparseEncoder networks from scratch. For more details, see [Training Overview](../../sparse_encoder/training_overview.md).
Note that models from `sentence_transformers.models` can be used to such as `sentence_transformers.models.Transformer` see [SentenceTransformer Models](../sentence_transformer/models.md)

## SPLADE Pooling
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.models.SpladePooling
   :members:
```

## MLM Transformer
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.models.MLMTransformer
   :members:
```

## CSR Sparsity
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.models.CSRSparsity
   :members:
```

## IDF
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.models.IDF
   :members:
``` 