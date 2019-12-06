# NLI Models
Conneau et al., 2017, show in the InferSent-Paper ([Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364)) that training on Natural Language Inference (NLI) data can produce universal sentence embeddings.

The datasets labeled sentence pairs with the labels *entail*, *contradict*, and *neutral*. For both sentences, we compute a sentence embedding. These two embeddings are concatenated and passed to softmax classifier to derive the final label.

As shown, this produces sentence embeddings that can be used for various use cases like clustering or semantic search.

# Datasets
We train the models on the [SNLI](https://nlp.stanford.edu/projects/snli/) and on the [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/) dataset. We call the combination of the two datasets AllNLI.

For a training example, see [examples/training_nli_bert.py](../../examples/training_nli_bert.py). 

# Pre-trained models
We provide the following pre-trained models. The performance was evaluated on the test set of the [STS benchmark dataset](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) using Spearman rank correlation.


### BERT models
- **bert-base-nli-mean-tokens**: BERT-base model with mean-tokens pooling. Performance: STSbenchmark: 77.12
- **bert-base-nli-max-tokens**: BERT-base with max-tokens pooling. Performance: STSbenchmark: 77.18
- **bert-base-nli-cls-token**: BERT-base with cls token pooling. Performance: STSbenchmark: 76.30
- **bert-large-nli-mean-tokens**: BERT-large with mean-tokens pooling. Performance: STSbenchmark: 79.19
- **bert-large-nli-max-tokens**: BERT-large with max-tokens pooling. Performance: STSbenchmark: 78.32
- **bert-large-nli-cls-token**: BERT-large with CLS token pooling. Performance: STSbenchmark: 78.29

### RoBERTa models
RoBERTa is an extension of BERT. [More Information](https://arxiv.org/abs/1907.11692).
- **roberta-base-nli-mean-tokens**: RoBERTa-base with mean-tokens pooling. Performance: STSbenchmark: 77.42
- **roberta-large-nli-mean-tokens**: RoBERTa-base with mean-tokens pooling. Performance: STSbenchmark: 78.58

### DistilBERT models
DistilBERT is a small, fast, cheap and light Transformer model based on Bert architecture. [More Information](https://github.com/huggingface/transformers/tree/master/examples/distillation)
- **distilbert-base-nli-mean-tokens**: DistilBERT-base with mean-tokens pooling. Performance: STSbenchmark: 76.97

# Performance Comparison
Here are the performances on the STS benchmark for other sentence embeddings methods. They were also computed by using cosine-similarity and Spearman rank correlation:
- Avg. GloVe embeddings:  58.02 
- BERT-as-a-service avg. embeddings:  46.35 
- BERT-as-a-service CLS-vector: 16.50 
- InferSent - GloVe: 68.03 
- Universal Sentence Encoder: 74.92

# Applications
This model works well in accessing the coarse-grained similarity between sentences. For application examples, see [examples/application_semantic_search.py](../../examples/application_semantic_search.py) and [examples/application_clustering.py](../../examples/application_clustering.py)