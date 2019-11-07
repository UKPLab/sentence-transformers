# NLI Models
Conneau et al., 2017, show in the InferSent-Paper ([Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364)) that training on Natural Language Inference (NLI) data can produce universal sentence embeddings.

The datasets labeled sentence pairs with the labels *entail*, *contradict*, and *neutral*. For both sentences, we compute a sentence embedding. These two embeddings are concatenated and passed to softmax classifier to derive the final label.

As shown, this produces sentence embeddings that can be used for various use cases like clustering or semantic search.

# Datasets
We train the models on the [SNLI](https://nlp.stanford.edu/projects/snli/) and on the [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/) dataset. We call the combination of the two datasets AllNLI.

For a training example, see [examples/training_nli_bert.py](../../examples/training_nli_bert.py). 

# Pre-trained models
 We provide the following pre-trained models:
 - **bert-base-nli-mean-tokens**: This model fine-tuned BERT-base on the AllNLI dataset. As pooling strategy, mean-tokens was used. Performance: STSbenchmark: 77.12
- **bert-base-nli-max-tokens**: This model fine-tuned BERT-base on the AllNLI dataset. As pooling strategy, max-tokens was used. Performance: STSbenchmark: 77.18
- **bert-large-nli-mean-tokens**: This model fine-tuned BERT-large on the AllNLI dataset. As pooling strategy, mean-tokens was used. Performance: STSbenchmark: 79.19
- **bert-large-nli-max-tokens**: This model fine-tuned BERT-large on the AllNLI dataset. As pooling strategy, max-tokens was used. Performance: STSbenchmark: 78.32

The performance was evaluated on the test set of the [STS benchmark dataset](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) by computing the cosine-similarity between two sentence embeddings and by computing the Spearman rank correlation to the gold labels.

# Performance Comparison
Here are the performances on the STS benchmark for other sentence embeddings methods. They were also computed by using cosine-similarity and Spearman rank correlation:
- Avg. GloVe embeddings:  58.02 
- BERT-as-a-service avg. embeddings:  46.35 
- BERT-as-a-service CLS-vector: 16.50 
- InferSent - GloVe: 68.03 
- Universal Sentence Encoder: 74.92

# Applications
This model works well in accessing the coarse-grained similarity between sentences. For application examples, see [examples/application_semantic_search.py](../../examples/application_semantic_search.py) and [examples/application_clustering.py](../../examples/application_clustering.py)