<!--- BADGES: START --->
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-models-yellow)](https://huggingface.co/models?library=sentence-transformers)
[![GitHub - License](https://img.shields.io/github/license/UKPLab/sentence-transformers?logo=github&style=flat&color=green)][#github-license]
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sentence-transformers?logo=pypi&style=flat&color=blue)][#pypi-package]
[![PyPI - Package Version](https://img.shields.io/pypi/v/sentence-transformers?logo=pypi&style=flat&color=orange)][#pypi-package]
[![Docs - GitHub.io](https://img.shields.io/static/v1?logo=github&style=flat&color=pink&label=docs&message=sentence-transformers)][#docs-package]
<!-- [![PyPI - Downloads](https://img.shields.io/pypi/dm/sentence-transformers?logo=pypi&style=flat&color=green)][#pypi-package] -->

[#github-license]: https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE
[#pypi-package]: https://pypi.org/project/sentence-transformers/
[#conda-forge-package]: https://anaconda.org/conda-forge/sentence-transformers
[#docs-package]: https://www.sbert.net/
<!--- BADGES: END --->

# Sentence Transformers: Multilingual Sentence, Paragraph, and Image Embeddings using BERT & Co.

This framework provides an easy method to compute dense vector representations for **sentences**, **paragraphs**, and **images**. The models are based on transformer networks like BERT / RoBERTa / XLM-RoBERTa etc. and achieve state-of-the-art performance in various tasks. Text is embedded in vector space such that similar text are closer and can efficiently be found using cosine similarity.

We provide an increasing number of **[state-of-the-art pretrained models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)** for more than 100 languages, fine-tuned for various use-cases.

Further, this framework allows an easy  **[fine-tuning of custom embeddings models](https://www.sbert.net/docs/sentence_transformer/training_overview.html)**, to achieve maximal performance on your specific task.

For the **full documentation**, see **[www.SBERT.net](https://www.sbert.net)**.

## Installation

We recommend **Python 3.8+**, **[PyTorch 1.11.0+](https://pytorch.org/get-started/locally/)**, and **[transformers v4.34.0+](https://github.com/huggingface/transformers)**.

**Install with pip**

```
pip install -U sentence-transformers
```

**Install with conda**

```
conda install -c conda-forge sentence-transformers
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

First download a pretrained model.

````python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
````

Then provide some sentences to the model.

````python
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# => (3, 384)
````

And that's already it. We now have a numpy arrays with the embeddings, one for each text. We can use these to compute similarities.

````python
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])
````

## Pre-Trained Models

We provide a large list of [Pretrained Models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) for more than 100 languages. Some models are general purpose models, while others produce embeddings for specific use cases. Pre-trained models can be loaded by just passing the model name: `SentenceTransformer('model_name')`.

## Training

This framework allows you to fine-tune your own sentence embedding methods, so that you get task-specific sentence embeddings. You have various options to choose from in order to get perfect sentence embeddings for your specific task. 

See [Training Overview](https://www.sbert.net/docs/sentence_transformer/training_overview.html) for an introduction how to train your own embedding models. We provide [various examples](https://github.com/UKPLab/sentence-transformers/tree/master/examples/training) how to train models on various datasets.

Some highlights are:
- Support of various transformer networks including BERT, RoBERTa, XLM-R, DistilBERT, Electra, BART, ...
- Multi-Lingual and multi-task learning
- Evaluation during training to find optimal model
- [20+ loss-functions](https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html) allowing to tune models specifically for semantic search, paraphrase mining, semantic similarity comparison, clustering, triplet loss, contrastive loss, etc.

## Application Examples

You can use this framework for:

- [Computing Sentence Embeddings](https://www.sbert.net/examples/applications/computing-embeddings/README.html)
- [Semantic Textual Similarity](https://www.sbert.net/docs/usage/semantic_textual_similarity.html)
- [Semantic Search](https://www.sbert.net/examples/applications/semantic-search/README.html)
- [Retrieve & Re-Rank](https://www.sbert.net/examples/applications/retrieve_rerank/README.html) 
- [Clustering](https://www.sbert.net/examples/applications/clustering/README.html)
- [Paraphrase Mining](https://www.sbert.net/examples/applications/paraphrase-mining/README.html)
- [Translated Sentence Mining](https://www.sbert.net/examples/applications/parallel-sentence-mining/README.html)
- [Multilingual Image Search, Clustering & Duplicate Detection](https://www.sbert.net/examples/applications/image-search/README.html)

and many more use-cases.

For all examples, see [examples/applications](https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications).

## Development setup

After cloning the repo (or a fork) to your machine, in a virtual environment, run:

```
python -m pip install -e ".[dev]"

pre-commit install
```

To test your changes, run:

```
pytest
```

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

Maintainer: [Tom Aarsen](https://github.com/tomaarsen), 🤗 Hugging Face

https://www.ukp.tu-darmstadt.de/

Don't hesitate to open an issue if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
