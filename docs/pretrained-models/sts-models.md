# STS Models
The models were first trained on [NLI data](nli-models.md), then we fine-tuned them on the  [STS benchmark dataset](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark). This generate sentence embeddings that are especially suitable to measure the semantic similarity between sentence pairs.

# Datasets
We use the training file from the  [STS benchmark dataset](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark).

For a training example, see:
- [examples/training_stsbenchmark.py](../../examples/training_stsbenchmark_bert.py) - Train directly on STS data
- [examples/training_stsbenchmark_continue_training.py ](../../examples/training_stsbenchmark_continue_training.py) - First train one NLI, than train on STS data.

# Pre-trained models
 We provide the following pre-trained models:
 
### BERT models
- **bert-base-nli-stsb-mean-tokens**: BERT-base trained on AllNLI, then on STS benchmark training set. Performance: STSbenchmark: 85.14
- **bert-large-nli-stsb-mean-tokens**: BERT-large trained on AllNLI, then on STS benchmark training set. Performance: STSbenchmark: 85.29

### RoBERTa models
RoBERTa is an extension of BERT. [More Information](https://arxiv.org/abs/1907.11692).
- **roberta-base-nli-stsb-mean-tokens**: RoBERTa-base trained on AllNLI, then on STS benchmark training set. Performance: STSbenchmark: 85.40
- **roberta-large-nli-stsb-mean-tokens**: RoBERTa-large trained on AllNLI, then on STS benchmark training set. Performance: STSbenchmark: 86.31

### DistilBERT
DistilBERT is a small, fast, cheap and light Transformer model based on Bert architecture. [More Information](https://github.com/huggingface/transformers/tree/master/examples/distillation)
- **distilbert-base-nli-stsb-mean-tokens**: Performance: STSbenchmark: 84.38

# Performance Comparison
Here are the performances on the STS benchmark for other sentence embeddings methods. They were also computed by using cosine-similarity and Spearman rank correlation. Note, these models were not-fined on the STS benchmark.

- Avg. GloVe embeddings:  58.02 
- BERT-as-a-service avg. embeddings:  46.35 
- BERT-as-a-service CLS-vector: 16.50 
- InferSent - GloVe: 68.03 
- Universal Sentence Encoder: 74.92
