SentenceTransformers Documentation
=================================================

SentenceTransformers is a Python framework for state-of-the-art sentence, text and image embeddings. The initial work is described in our paper `Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks <https://arxiv.org/abs/1908.10084>`_.

You can use this framework to compute sentence / text embeddings for more than 100 languages. These embeddings can then be compared e.g. with cosine-similarity to find sentences with a similar meaning. This can be useful for `semantic textual similar <docs/usage/semantic_textual_similarity.html>`_, `semantic search <examples/applications/semantic-search/README.html>`_, or `paraphrase mining <examples/applications/paraphrase-mining/README.html>`_.

The framework is based on `PyTorch <https://pytorch.org/>`_ and `Transformers <https://huggingface.co/transformers/>`_ and offers a large collection of `pre-trained models <docs/pretrained_models.html>`_ tuned for various tasks. Further, it is easy to `fine-tune your own models <docs/training/overview.html>`_.


Installation
=================================================

You can install it using pip:

.. code-block:: python

   pip install -U sentence-transformers


We recommand **Python 3.6** or higher, and at least **PyTorch 1.6.0**. See `installation <docs/installation.html>`_ for further installation options, especially if you want to use a GPU.



Usage
=================================================
The usage is as simple as:

.. code-block:: python

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')

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

Our models are evaluated extensively and achieve state-of-the-art performance on various tasks. Further, the code is tuned to provide the highest possible speed.


.. raw:: html

    <table class="docutils">
    <thead>
    <tr>
    <th>Model</th>
    <th align="center">STS benchmark</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>Avg. GloVe embeddings</td>
    <td align="center">58.02</td>
    </tr>
    <tr>
    <td>BERT-as-a-service avg. embeddings</td>
    <td align="center">46.35</td>
    </tr>
    <tr>
    <td>BERT-as-a-service CLS-vector</td>
    <td align="center">16.50</td>
    </tr>
    <tr>
    <td>InferSent - GloVe</td>
    <td align="center">68.03</td>
    </tr>
    <tr>
    <td>Universal Sentence Encoder</td>
    <td align="center">74.92</td>
    </tr>
    <tr>
    <td><strong>Sentence Transformer Models (NLI + MNLI)</strong></td>
    <td align="center"></td>
    </tr>
    <tr>
    <td>nli-distilroberta-base-v2</td>
    <td align="center">84.38</td>
    </tr>
    <tr>
    <td>nli-roberta-base-v2</td>
    <td align="center">85.54</td>
    </tr>
    <tr>
    <td>nli-mpnet-base-v2</td>
    <td align="center">86.53</td>
    </tr>
    <tr>
    <td><strong>Sentence Transformer Models (NLI + STS benchmark)</strong></td>
    <td align="center"></td>
    </tr>
    <tr>
    <td>stsb-distilroberta-base-v2</td>
    <td align="center">86.41</td>
    </tr>
    <tr>
    <td>stsb-roberta-base-v2</td>
    <td align="center">87.21</td>
    </tr>
    <tr>
    <td>stsb-mpnet-base-v2</td>
    <td align="center">88.57</td>
    </tr>
    </tbody>
    </table>




Contact
=========================

Contact person: Nils Reimers, reimers@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

*This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.*


Citing & Authors
=========================

If you find this repository helpful, feel free to cite our publication `Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks <https://arxiv.org/abs/1908.10084>`_:

 .. code-block:: javascript

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

 .. code-block:: javascript

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

 .. code-block:: javascript

  @article{thakur-2020-AugSBERT,
    title = "Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks",
    author = "Thakur, Nandan and Reimers, Nils and Daxenberger, Johannes and  Gurevych, Iryna",
    journal= "arXiv preprint arXiv:2010.08240",
    month = "10",
    year = "2020",
    url = "https://arxiv.org/abs/2010.08240",
  }



.. toctree::
   :maxdepth: 2
   :caption: Overview

   docs/installation
   docs/quickstart
   docs/pretrained_models
   docs/pretrained_cross-encoders
   docs/publications

.. toctree::
   :maxdepth: 2
   :caption: Usage

   examples/applications/computing-embeddings/README
   docs/usage/semantic_textual_similarity
   examples/applications/clustering/README
   examples/applications/paraphrase-mining/README
   examples/applications/parallel-sentence-mining/README
   examples/applications/semantic-search/README
   examples/applications/retrieve_rerank/README
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
   examples/training/quora_duplicate_questions/README
   examples/training/ms_marco/README

.. toctree::
   :maxdepth: 2
   :caption: Unsupervised Learning

   examples/unsupervised_learning/README
   examples/unsupervised_learning/tsdae/README
   examples/unsupervised_learning/SimCSE/README
   examples/unsupervised_learning/CT/README
   examples/unsupervised_learning/CT_In-Batch_Negatives/README
   examples/unsupervised_learning/MLM/README
   examples/unsupervised_learning/query_generation/README

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
