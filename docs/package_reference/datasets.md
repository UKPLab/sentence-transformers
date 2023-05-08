# Datasets
`sentence_transformers.datasets` contains classes to organize your training input examples.



## ParallelSentencesDataset
`ParallelSentencesDataset` is used for multilingual training. For details, see [multilingual training](../../examples/training/multilingual/README.md).
```eval_rst
.. autoclass:: sentence_transformers.datasets.ParallelSentencesDataset
```


## SentenceLabelDataset
`SentenceLabelDataset` can be used if you have labeled sentences and want to train with triplet loss.
```eval_rst
.. autoclass:: sentence_transformers.datasets.SentenceLabelDataset
```

## DenoisingAutoEncoderDataset
`DenoisingAutoEncoderDataset` is used for unsupervised training with the TSDAE method.
```eval_rst
.. autoclass:: sentence_transformers.datasets.DenoisingAutoEncoderDataset
```

## NoDuplicatesDataLoader
`NoDuplicatesDataLoader`can be used together with MultipleNegativeRankingLoss to ensure that no duplicates are within the same batch.
```eval_rst
.. autoclass:: sentence_transformers.datasets.NoDuplicatesDataLoader
```


