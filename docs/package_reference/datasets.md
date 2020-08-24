# Datasets
`sentence_transformers.datasets` contains classes to organize your training input examples.


## SentencesDataset
`SentencesDataset` is the main class to store training classes for training. For details, see [training overview](../training/overview.md). 
```eval_rst
.. autoclass:: sentence_transformers.datasets.SentencesDataset
```

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
