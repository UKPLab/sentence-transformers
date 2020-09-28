SentenceTransformers Documentation
=================================================

SentenceTransformers is a Python framework for state-of-the-art sentence and text embeddings. The initial work is described in our paper `Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks <https://arxiv.org/abs/1908.10084>`_.

You can use this framework to compute sentence / text embeddings for more than 100 languages. These embeddings can then be compared e.g. with cosine-similarity to find sentences with a similar meaning. This can be useful for `semantic textual similar <docs/usage/semantic_textual_similarity.html>`_, `semantic search <docs/usage/semantic_search.html>`_, or `paraphrase mining <docs/usage/paraphrase_mining.html>`_.

The framework is based on `PyTorch <https://pytorch.org/>`_ and `Transformers <https://huggingface.co/transformers/>`_ and offers a large collection of `pre-trained models <docs/pretrained_models.html>`_ tuned for various tasks. Further, it is easy to `fine-tune your own models <docs/training/overview.html>`_.


Installation
=================================================

You can install it using pip:

.. code-block:: python

   pip install -U sentence-transformers


We recommand **Python 3.6** or higher, and at least **PyTorch 1.2.0**. PyTorch 1.6.0 or higher is recommended and needed for some features. See `installation <docs/installation.html>`_ for further installation options, especially if you want to use a GPU.



Usage
=================================================
The usage is as simple as:

.. code-block:: python

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

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
    <th align="center">SentEval</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>Avg. GloVe embeddings</td>
    <td align="center">58.02</td>
    <td align="center">81.52</td>
    </tr>
    <tr>
    <td>BERT-as-a-service avg. embeddings</td>
    <td align="center">46.35</td>
    <td align="center">84.04</td>
    </tr>
    <tr>
    <td>BERT-as-a-service CLS-vector</td>
    <td align="center">16.50</td>
    <td align="center">84.66</td>
    </tr>
    <tr>
    <td>InferSent - GloVe</td>
    <td align="center">68.03</td>
    <td align="center">85.59</td>
    </tr>
    <tr>
    <td>Universal Sentence Encoder</td>
    <td align="center">74.92</td>
    <td align="center">85.10</td>
    </tr>
    <tr>
    <td><strong>Sentence Transformer Models</strong></td>
    <td align="center"></td>
    <td align="center"></td>
    </tr>
    <tr>
    <td>bert-base-nli-mean-tokens</td>
    <td align="center">77.12</td>
    <td align="center">86.37</td>
    </tr>
    <tr>
    <td>bert-large-nli-mean-tokens</td>
    <td align="center">79.19</td>
    <td align="center">87.78</td>
    </tr>
    <tr>
    <td>bert-base-nli-stsb-mean-tokens</td>
    <td align="center">85.14</td>
    <td align="center">86.07</td>
    </tr>
    <tr>
    <td>bert-large-nli-stsb-mean-tokens</td>
    <td align="center">85.29</td>
    <td align="center">86.66</td>
    </tr>
    <tr>
    <td>roberta-base-nli-stsb-mean-tokens</td>
    <td align="center">85.44</td>
    <td align="center">-</td>
    </tr>
    <tr>
    <td>roberta-large-nli-stsb-mean-tokens</td>
    <td align="center">86.39</td>
    <td align="center">-</td>
    </tr>
    <tr>
    <td>distilbert-base-nli-stsb-mean-tokens</td>
    <td align="center">85.16</td>
    <td align="center">-</td>
    </tr>
    </tbody>
    </table>



.. toctree::
   :maxdepth: 2
   :caption: Overview

   docs/installation
   docs/quickstart
   docs/pretrained_models
   docs/publications

.. toctree::
   :maxdepth: 2
   :caption: Usage

   docs/usage/computing_sentence_embeddings
   docs/usage/semantic_textual_similarity
   docs/usage/paraphrase_mining
   docs/usage/semantic_search


.. toctree::
   :maxdepth: 2
   :caption: Training

   docs/training/overview
   examples/training/multilingual/README
   examples/training/distillation/README

.. toctree::
   :maxdepth: 2
   :caption: Training Examples

   examples/training/sts/README
   examples/training/nli/README
   examples/training/quora_duplicate_questions/README


.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   docs/package_reference/SentenceTransformer
   docs/package_reference/util
   docs/package_reference/models
   docs/package_reference/losses
   docs/package_reference/evaluation
   docs/package_reference/datasets
