# Sentence Transformers: Multilingual Sentence, Paragraph, and Image Embeddings using BERT & Co.

This framework provides an easy method to compute dense vector representations for **sentences**, **paragraphs**, and **images**. The models are based on transformer networks like BERT / RoBERTa / XLM-RoBERTa etc. and achieve state-of-the-art performance in various task. Text is embedding in vector space such that similar text is close and can efficiently be found using cosine similarity.


We provide an increasing number of **[state-of-the-art pretrained models](https://www.sbert.net/docs/pretrained_models.html)** for more than 100 languages, fine-tuned for various use-cases.

Further, this framework allows an easy  **[fine-tuning of custom embeddings models](https://www.sbert.net/docs/training/overview.html)**, to achieve maximal performance on your specific task.


For the **full documentation**, see **[www.SBERT.net](https://www.sbert.net)**.

The following publications are integrated in this framework:
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) (EMNLP 2019)
- [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813) (EMNLP 2020)
- [Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks](https://arxiv.org/abs/2010.08240) (NAACL 2021)
- [The Curse of Dense Low-Dimensional Information Retrieval for Large Index Sizes](https://arxiv.org/abs/2012.14210) (arXiv 2020)
- [TSDAE: Using Transformer-based Sequential Denoising Auto-Encoderfor Unsupervised Sentence Embedding Learning](https://arxiv.org/abs/2104.06979) (arXiv 2021)
- [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663) (arXiv 2021)


## Installation
We recommend **Python 3.6** or higher, **[PyTorch 1.6.0](https://pytorch.org/get-started/locally/)** or higher and **[transformers v3.1.0](https://github.com/huggingface/transformers)** or higher. The code does **not** work with Python 2.7.




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

**PyTorch with CUDA**
If you want to use a GPU / CUDA, you must install PyTorch with the matching CUDA Version. Follow
[PyTorch - Get Started](https://pytorch.org/get-started/locally/) for further details how to install PyTorch.



## Getting Started

See [Quickstart](https://www.sbert.net/docs/quickstart.html) in our documenation.


[This example](https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/computing-embeddings/computing_embeddings.py) shows you how to use an already trained Sentence Transformer model to embed sentences for another task.

First download a pretrained model.
````python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-distilroberta-base-v1')
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
- Support of various transformer networks including BERT, RoBERTa, XLM-R, DistilBERT, Electra, BART, ...
- Multi-Lingual and multi-task learning
- Evaluation during training to find optimal model
- [10+ loss-functions](https://www.sbert.net/docs/package_reference/losses.html) allowing to tune models specifically for semantic search, paraphrase mining, semantic similarity comparison, clustering, triplet loss, constrative loss.



## Performance

Our models are evaluated extensively and achieve state-of-the-art performance on various tasks. Further, the code is tuned to provide the highest possible speed.

| Model    | STS benchmark | 
| ----------------------------------|:-----: |
| Avg. GloVe embeddings             | 58.02  | 
| BERT-as-a-service avg. embeddings | 46.35  | 
| BERT-as-a-service CLS-vector      | 16.50  | 
| InferSent - GloVe                 | 68.03  |
| Universal Sentence Encoder        | 74.92  | 
|**Sentence Transformer Models (NLI + MNLI)**  | |
| nli-distilroberta-base-v2 | 84.38 |
| nli-roberta-base-v2 | 85.54 |
| nli-mpnet-base-v2 | 86.53 |
|**Sentence Transformer Models (NLI + STS benchmark)**  | |
| stsb-distilroberta-base-v2 | 86.41 |
| stsb-roberta-base-v2 | 87.21 |
| stsb-mpnet-base-v2 | 88.57 |



[» Full list of pretrained models](https://www.sbert.net/docs/pretrained_models.html)




## Application Examples
You can use this framework for:
- [Computing Sentence Embeddings](https://www.sbert.net/examples/applications/computing-embeddings/README.html)
- [Semantic Textual Similarity](https://www.sbert.net/docs/usage/semantic_textual_similarity.html)
- [Clustering](https://www.sbert.net/examples/applications/clustering/README.html)
- [Paraphrase Mining](https://www.sbert.net/examples/applications/paraphrase-mining/README.html)
 - [Translated Sentence Mining](https://www.sbert.net/examples/applications/parallel-sentence-mining/README.html)
 - [Semantic Search](https://www.sbert.net/examples/applications/semantic-search/README.html)
 - [Retrieve & Re-Rank](https://www.sbert.net/examples/applications/retrieve_rerank/README.html) 
 - [Text Summarization](https://www.sbert.net/examples/applications/text-summarization/README.html) 
- [Multilingual Image Search, Clustering & Duplicate Detection](https://www.sbert.net/examples/applications/image-search/README.html)

and many more use-cases.


For all examples, see [examples/applications](https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications).

## Citing & Authors
If you find this repository helpful, feel free to cite our publication [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084):
```bibtex 
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```


If you use one of the multilingual models, feel free to cite our publication [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813):
```bibtex
@inproceedings{reimers-2020-multilingual-sentence-bert,
    title = "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2004.09813",
}
```

Please have a look at [Publications](https://www.sbert.net/docs/publications.html) for our different publications that are integrated into SentenceTransformers.


Contact person: [Nils Reimers](https://www.nils-reimers.de), [info@nils-reimers.de](mailto:info@nils-reimers.de)

https://www.ukp.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.







