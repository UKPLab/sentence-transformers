# Models
`sentence_transformers.models` defines different building blocks, a.k.a. Modules, that can be used to create SentenceTransformer models from scratch. For more details, see [Creating Custom Models](../../sentence_transformer/usage/custom_models.rst).

## Main Modules
```{eval-rst}
.. autoclass:: sentence_transformers.models.Transformer
.. autoclass:: sentence_transformers.models.Pooling
.. autoclass:: sentence_transformers.models.Dense
```

## Further Modules
```{eval-rst}
.. autoclass:: sentence_transformers.models.Asym
.. autoclass:: sentence_transformers.models.BoW
.. autoclass:: sentence_transformers.models.CNN
.. autoclass:: sentence_transformers.models.LSTM
.. autoclass:: sentence_transformers.models.Normalize
.. autoclass:: sentence_transformers.models.StaticEmbedding
    :members: from_model2vec, from_distillation
.. autoclass:: sentence_transformers.models.WeightedLayerPooling
.. autoclass:: sentence_transformers.models.WordEmbeddings
.. autoclass:: sentence_transformers.models.WordWeights
```

## Base Modules
```{eval-rst}
.. autoclass:: sentence_transformers.models.Module
.. autoclass:: sentence_transformers.models.InputModule
```