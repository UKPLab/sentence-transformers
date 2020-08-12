# STS Models
The models were first trained on [NLI data](nli-models.md), then we fine-tuned them on the  [STS benchmark dataset](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark). This generate sentence embeddings that are especially suitable to measure the semantic similarity between sentence pairs.

# Datasets
We use the training file from the  [STS benchmark dataset](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark).

For a training example, see:
- [examples/training_stsbenchmark.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/training_transformers/training_stsbenchmark.py) - Train directly on STS data
- [examples/training_stsbenchmark_continue_training.py ](https://github.com/UKPLab/sentence-transformers/blob/master/examples/training_transformers/training_stsbenchmark_continue_training.py) - First train on NLI, than train on STS data.

# Pre-trained models
 We provide the following pre-trained models:

[» Full List of STS Models](https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0)

# Performance Comparison
Here are the performances on the STS benchmark for other sentence embeddings methods. They were also computed by using cosine-similarity and Spearman rank correlation. Note, these models were not-fined on the STS benchmark.

- Avg. GloVe embeddings:  58.02 
- BERT-as-a-service avg. embeddings:  46.35 
- BERT-as-a-service CLS-vector: 16.50 
- InferSent - GloVe: 68.03 
- Universal Sentence Encoder: 74.92
