# Sentence Transformers: Multilingual Sentence Embeddings using BERT / RoBERTa / XLM-RoBERTa & Co. with PyTorch

This framework provides an easy method to compute dense vector representations for sentences and paragraphs (also known as sentence embeddings). The models are based on transformer networks like BERT / RoBERTa / XLM-RoBERTa etc. and are tuned specificially meaningul sentence embeddings such that sentences with similar meanings are close in vector space.


We provide an increasing number of **[state-of-the-art pretrained models](https://www.sbert.net/docs/pretrained_models.html)** for more than 100 languages, fine-tuned for various use-cases.

Further, this framework allows an easy  **[fine-tuning of custom embeddings models](https://www.sbert.net/docs/training/overview.html)**, to achieve maximal performance on your specific task.


For the **full documentation**, see [www.SBERT.net](https://www.sbert.net), as well as our publications:
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) (EMNLP 2019)
- [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813) (ArXiv 2020).





## Installation
We recommend **Python 3.6** or higher, **[PyTorch 1.2.0](https://pytorch.org/get-started/locally/)** or higher and **[transformers v3.0.2](https://github.com/huggingface/transformers)** or higher. The code does **not** work with Python 2.7.

First, follow the installation for PyTroch you can find here: [PyTorch - Get Started](https://pytorch.org/get-started/locally/). As the provided models can have a high computational overhead, it is recommend to run them on a GPU. See the PyTorch page how to install PyTorch for GPU (CUDA).


**Install with pip**

Install the *sentence-transformers* with `pip`:
```
pip install -U sentence-transformers
```

**Install from sources**

Alternatively, you can also clone the latest version from the [repository](https://github.com/UKPLab/sentence-transformers) and install it directly from the source code:
````
pip install -e .
```` 



## Getting Started

See [Quickstart](https://www.sbert.net/docs/quickstart.html) in our documenation.


[This example](https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/computing_embeddings.py) shows you how to use an already trained Sentence Transformer model to embed sentences for another task.

First download a pretrained model.
````python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
````
Then provide some sentences to the model.
````python
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']
sentence_embeddings = model.encode(sentences)
````
And that's it already. We now have a list of numpy arrays with the embeddings.
````python
for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
````

## Pre-Trained Models

We provide a large list of [Pretrained Models](https://www.sbert.net/docs/pretrained_models.html) for more than 100 languages. Some models are general purpose models, while others produce embeddings for specific use cases. Pre-trained models can be loaded by just passing the model name: `SentenceTransformer('model_name')`.

[»  Full list of pretrained models](https://www.sbert.net/docs/pretrained_models.html)



## Training
This framework allows you to fine-tune your own sentence embedding methods, so that you get task-specific sentence embeddings. You have various options to choose from in order to get perfect sentence embeddings for your specific task. 

See [Training Overview](https://www.sbert.net/docs/training/overview.html) for an introduction how to train your own embedding models. We provide [various examples](https://github.com/UKPLab/sentence-transformers/tree/master/examples/training) how to train models on various datasets.


Some highlights are:
- Support of various transformer networks including BERT, RoBERTa, XLM-R, DistilBERT, Elextra, BART, ...
- Multi-Lingual and multi-task learning
- Evaluation during training to find optimal model
- [10+ loss-functions](https://www.sbert.net/docs/package_reference/losses.html) allowing to tune models specifically for semantic search, paraphrase mining, semantic similarity comparison, clustering, triplet loss, constrative loss.








## Performance

Extensive evaluation is currently undergoing, but here we provide some preliminary results.

| Model    | STS benchmark | SentEval  |
| ----------------------------------|:-----: |:---:   |
| Avg. GloVe embeddings             | 58.02  | 81.52  |
| BERT-as-a-service avg. embeddings | 46.35  | 84.04  |
| BERT-as-a-service CLS-vector      | 16.50  | 84.66  |
| InferSent - GloVe                 | 68.03  | 85.59  |
| Universal Sentence Encoder        | 74.92  | 85.10  |
|**Sentence Transformer Models**    ||
| bert-base-nli-mean-tokens         | 77.12  | 86.37 |
| bert-large-nli-mean-tokens        | 79.19  | 87.78 |
| bert-base-nli-stsb-mean-tokens    | 85.14  | 86.07 |
| bert-large-nli-stsb-mean-tokens   | 85.29 | 86.66|
| roberta-base-nli-stsb-mean-tokens | 85.44 | - |
| roberta-large-nli-stsb-mean-tokens | 86.39 | - |
| distilbert-base-nli-stsb-mean-tokens | 85.16 | - |





## Application Examples
We present some examples, how the generated sentence embeddings can be used for downstream applications.

### Semantic Search
Semantic search is the task of finding similar sentences to a given sentence. See [semantic_search.py](https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/semantic_search.py) for an example. See [our documentation](https://www.sbert.net/docs/usage/semantic_search.html) on more details about semantic search.

We first generate an embedding for all sentences in a corpus:
```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Corpus with example sentences
corpus = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'A cheetah is running behind its prey.'
          ]
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Query sentences:
queries = ['A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.', 'A cheetah chases prey on across a field.']


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = 5
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()

    #We use np.argpartition, to only partially sort the top_k results
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for idx in top_results[0:top_k]:
        print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))
```

The output looks like this:
```
Query: A man is eating pasta.
Top 5 most similar sentences in corpus:
A man is eating a piece of bread. (Score: 0.8518)
A man is eating a food. (Score: 0.8020)
A monkey is playing drums. (Score: 0.4167)
A man is riding a horse. (Score: 0.2621)
A man is riding a white horse on an enclosed ground. (Score: 0.2379)


Query: Someone in a gorilla costume is playing a set of drums.
Top 5 most similar sentences in corpus:
A monkey is playing drums. (Score: 0.8514)
A man is eating a piece of bread. (Score: 0.3671)
A man is eating a food. (Score: 0.3559)
A man is riding a horse. (Score: 0.3153)
The girl is carrying a baby. (Score: 0.2589)


Query: A cheetah chases prey on across a field.
Top 5 most similar sentences in corpus:
A cheetah is running behind its prey. (Score: 0.9073)
Two men pushed carts through the woods. (Score: 0.3896)
A man is riding a horse. (Score: 0.3789)
A man is riding a white horse on an enclosed ground. (Score: 0.3544)
A monkey is playing drums. (Score: 0.3435)

```


### Clustering
[clustering.py](https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/clustering.py) depicts an example to cluster similar sentences based on their sentence embedding similarity.

As before, we first compute an embedding for each sentence:
```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')


# Corpus with example sentences
corpus = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'A man is eating pasta.',
          'The girl is carrying a baby.',
          'The baby is carried by the woman',
          'A man is riding a horse.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'Someone in a gorilla costume is playing a set of drums.',
          'A cheetah is running behind its prey.',
          'A cheetah chases prey on across a field.']

corpus_embeddings = embedder.encode(corpus)


# Then, we perform k-means clustering using sklearn:
from sklearn.cluster import KMeans

num_clusters = 5
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_
```

The output looks like this:
```
Cluster  1
['The girl is carrying a baby.', 'The baby is carried by the woman']

Cluster  2
['A cheetah is running behind its prey.', 'A cheetah chases prey on across a field.']

Cluster  3
['A monkey is playing drums.', 'Someone in a gorilla costume is playing a set of drums.']

Cluster  4
['A man is eating food.', 'A man is eating a piece of bread.', 'A man is eating pasta.']

Cluster  5
['A man is riding a horse.', 'A man is riding a white horse on an enclosed ground.']
```


For more examples, see [examples/applications](https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications).

## Citing & Authors
If you find this repository helpful, feel free to cite our publication [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084):
``` 
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/1908.10084",
}
```


If you use the code for multilingual models, feel free to cite our publication [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813):
``` 
@article{reimers-2020-multilingual-sentence-bert,
    title = "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation",
    author = "Reimers, Nils and Gurevych, Iryna",
    journal= "arXiv preprint arXiv:2004.09813",
    month = "04",
    year = "2020",
    url = "http://arxiv.org/abs/2004.09813",
}
```


The main contributors of this repository are:
- [Nils Reimers](https://github.com/nreimers)
- [Gregor Geigle](https://github.com/aaronsom)

Contact person: Nils Reimers, info@nils-reimers.de

https://www.ukp.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.







