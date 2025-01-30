.. tip::

   You can now use ONNX and OpenVINO backends for Sentence Transformer models. Read `SentenceTransformer > Usage > Speeding up Inference <docs/sentence_transformer/usage/efficiency.html>`_ to learn more about the new backends and what they can mean for your inference speed.

.. tip::

   Sentence Transformers v3.4 just released, introducing documentation for training with PEFT. Read `SentenceTransformer > Training Examples > Training with PEFT Adapters <examples/training/peft/README.html>`_ to learn more about how you can use train embedding models without finetuning all model parameters.

SentenceTransformers Documentation
==================================

Sentence Transformers (a.k.a. SBERT) is the go-to Python module for accessing, using, and training state-of-the-art text and image embedding models.
It can be used to compute embeddings using Sentence Transformer models (`quickstart <docs/quickstart.html#sentence-transformer>`_) or to calculate similarity scores using Cross-Encoder models (`quickstart <docs/quickstart.html#cross-encoder>`_). This unlocks a wide range of applications, including `semantic search <examples/applications/semantic-search/README.html>`_, `semantic textual similarity <docs/usage/semantic_textual_similarity.html>`_, and `paraphrase mining <examples/applications/paraphrase-mining/README.html>`_.

A wide selection of over `5,000 pre-trained Sentence Transformers models <https://huggingface.co/models?library=sentence-transformers>`_ are available for immediate use on 🤗 Hugging Face, including many of the state-of-the-art models from the `Massive Text Embeddings Benchmark (MTEB) leaderboard <https://huggingface.co/spaces/mteb/leaderboard>`_. Additionally, it is easy to `train or finetune your own models <docs/sentence_transformer/training_overview.html>`_ using Sentence Transformers, enabling you to create custom models for your specific use cases.

Sentence Transformers was created by `UKPLab <http://www.ukp.tu-darmstadt.de/>`_ and is being maintained by `🤗 Hugging Face <https://huggingface.co>`_. Don't hesitate to open an issue on the `Sentence Transformers repository <https://github.com/UKPLab/sentence-transformers>`_ if something is broken or if you have further questions.

Usage
=====
.. seealso::
  
   See the `Quickstart <docs/quickstart.html>`_ for more quick information on how to use Sentence Transformers.

Using Sentence Transformer models is elementary:

.. sidebar:: Installation

   You can install *sentence-transformers* using pip:
   
   .. code-block:: python
   
      pip install -U sentence-transformers
   
   We recommend **Python 3.9+** and **PyTorch 1.11.0+**. See `installation <docs/installation.html>`_ for further installation options.

.. code-block:: python

   from sentence_transformers import SentenceTransformer

   # 1. Load a pretrained Sentence Transformer model
   model = SentenceTransformer("all-MiniLM-L6-v2")

   # The sentences to encode
   sentences = [
       "The weather is lovely today.",
       "It's so sunny outside!",
       "He drove to the stadium.",
   ]

   # 2. Calculate embeddings by calling model.encode()
   embeddings = model.encode(sentences)
   print(embeddings.shape)
   # [3, 384]

   # 3. Calculate the embedding similarities
   similarities = model.similarity(embeddings, embeddings)
   print(similarities)
   # tensor([[1.0000, 0.6660, 0.1046],
   #         [0.6660, 1.0000, 0.1411],
   #         [0.1046, 0.1411, 1.0000]])

What Next?
==========

Consider reading one of the following sections to answer the related questions:

* How to **use** Sentence Transformer models? `Sentence Transformers > Usage <docs/sentence_transformer/usage/usage.html>`_
* What Sentence Transformer **models** can I use? `Sentence Transformers > Pretrained Models <docs/sentence_transformer/pretrained_models.html>`_
* How do I make Sentence Transformer models **faster**? `Sentence Transformers > Usage > Speeding up Inference <docs/sentence_transformer/usage/efficiency.html>`_
* How do I **train/finetune** a Sentence Transformer model? `Sentence Transformers > Training Overview <docs/sentence_transformer/training_overview.html>`_
* How to **use** Cross Encoder models? `Cross Encoder > Usage <docs/cross_encoder/usage/usage.html>`_
* What Cross Encoder **models** can I use? `Cross Encoder > Pretrained Models <docs/cross_encoder/pretrained_models.html>`_

Citing
======

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
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   docs/installation
   docs/quickstart

.. toctree::
   :maxdepth: 2
   :caption: Sentence Transformer
   :hidden:

   docs/sentence_transformer/usage/usage
   docs/sentence_transformer/pretrained_models
   docs/sentence_transformer/training_overview
   docs/sentence_transformer/dataset_overview
   docs/sentence_transformer/loss_overview
   docs/sentence_transformer/training/examples

.. toctree::
   :maxdepth: 2
   :caption: Cross Encoder
   :hidden:

   docs/cross_encoder/usage/usage
   docs/cross_encoder/pretrained_models
   docs/cross_encoder/training_overview
   docs/cross_encoder/training/examples

.. toctree::
   :maxdepth: 3
   :caption: Package Reference
   :glob:
   :hidden:

   docs/package_reference/sentence_transformer/index
   docs/package_reference/cross_encoder/index
   docs/package_reference/util
