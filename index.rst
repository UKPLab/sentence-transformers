SentenceTransformers Documentation
=================================================

SentenceTransformers is a Python framework for state-of-the-art sentence, text and image embeddings. The initial work is described in our paper `Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks <https://arxiv.org/abs/1908.10084>`_.

You can use this framework to compute sentence / text embeddings for more than 100 languages. These embeddings can then be compared e.g. with cosine-similarity to find sentences with a similar meaning. This can be useful for `semantic textual similarity <docs/usage/semantic_textual_similarity.html>`_, `semantic search <examples/applications/semantic-search/README.html>`_, or `paraphrase mining <examples/applications/paraphrase-mining/README.html>`_.

The framework is based on `PyTorch <https://pytorch.org/>`_ and `Transformers <https://huggingface.co/transformers/>`_ and offers a large collection of `pre-trained models <docs/pretrained_models.html>`_ tuned for various tasks. Further, it is easy to `fine-tune your own models <docs/training/overview.html>`_.


Installation
=================================================

You can install it using pip:

.. code-block:: python

   pip install -U sentence-transformers


We recommend **Python 3.6** or higher, and at least **PyTorch 1.6.0**. See `installation <docs/installation.html>`_ for further installation options, especially if you want to use a GPU.



Usage
=================================================
The usage is as simple as:

.. code-block:: python

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    #Our sentences we like to encode
    sentences = ['This framework generates embeddings for each input sentence',
        'Sentences are passed as a list of string.',
        'The quick brown fox jumps over the lazy dog.']

    #Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)

    #Print the embeddings
    for sentence, embedding in zip(sentences, embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("")




Performance
=========================

Our models are evaluated extensively and achieve state-of-the-art performance on various tasks. Further, the code is tuned to provide the highest possible speed. Have a look at `Pre-Trained Models <https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/>`_ for an overview of available models and the respective performance on different tasks.






Contact
=========================

Contact person: Nils Reimers, info@nils-reimers.de

https://www.ukp.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

*This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.*


Citing & Authors
=========================

If you find this repository helpful, feel free to cite our publication `Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks <https://arxiv.org/abs/1908.10084>`_:

 .. code-block:: bibtex

  @inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
  }



If you use one of the multilingual models, feel free to cite our publication `Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation <https://arxiv.org/abs/2004.09813>`_:

 .. code-block:: bibtex

  @inproceedings{reimers-2020-multilingual-sentence-bert,
    title = "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2004.09813",
  }



If you use the code for `data augmentation <https://github.com/UKPLab/sentence-transformers/tree/master/examples/training/data_augmentation>`_, feel free to cite our publication `Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks <https://arxiv.org/abs/2010.08240>`_:

 .. code-block:: bibtex

  @inproceedings{thakur-2020-AugSBERT,
    title = "Augmented {SBERT}: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks",
    author = "Thakur, Nandan and Reimers, Nils and Daxenberger, Johannes  and Gurevych, Iryna",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.28",
    pages = "296--310",
  }



.. toctree::
   :maxdepth: 2
   :caption: Overview

   docs/installation
   docs/quickstart
   docs/pretrained_models
   docs/pretrained_cross-encoders
   docs/publications
   docs/hugging_face

.. toctree::
   :maxdepth: 2
   :caption: Usage

   examples/applications/computing-embeddings/README
   docs/usage/semantic_textual_similarity
   examples/applications/semantic-search/README
   examples/applications/retrieve_rerank/README
   examples/applications/clustering/README
   examples/applications/paraphrase-mining/README
   examples/applications/parallel-sentence-mining/README
   examples/applications/cross-encoder/README
   examples/applications/image-search/README

.. toctree::
   :maxdepth: 2
   :caption: Training

   docs/training/overview
   examples/training/multilingual/README
   examples/training/distillation/README
   examples/training/cross-encoder/README
   examples/training/data_augmentation/README

.. toctree::
   :maxdepth: 2
   :caption: Training Examples

   examples/training/sts/README
   examples/training/nli/README
   examples/training/paraphrases/README
   examples/training/quora_duplicate_questions/README
   examples/training/ms_marco/README

.. toctree::
   :maxdepth: 2
   :caption: Unsupervised Learning

   examples/unsupervised_learning/README
   examples/domain_adaptation/README

.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   docs/package_reference/SentenceTransformer
   docs/package_reference/util
   docs/package_reference/models
   docs/package_reference/losses
   docs/package_reference/evaluation
   docs/package_reference/datasets
   docs/package_reference/cross_encoder
