SentenceTransformers Documentation
=================================================

SentenceTransformers is a Python framework for state-of-the-art sentence and text embeddings. The initial work is described in our paper `Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks <https://arxiv.org/abs/1908.10084>`_.

You can use this framework to compute sentence / text embeddings for more than 100 languages. These embeddings can then be compared e.g. with cosine-similarity to find sentences with a similar meaning. This can be useful for `semantic textual similar <docs/usage/semantic_textual_similarity.html>`_, `semantic search <docs/usage/semantic_search.html>`_, or `paraphrase mining <docs/usage/paraphrase_mining.html>`_.

The framework is based on `PyTorch <https://pytorch.org/>`_ and `Transformers <https://huggingface.co/transformers/>`_ and offers a large collection of `pre-trained models <docs/pretrained_models.html>`_ tuned for various tasks. Further, it is easy to `fine-tune your own models <docs/training/overview.html>`_.

After the `installation <docs/installation.html>`_, the usage is as simple as:

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
   :caption: Training Examples

   docs/training/overview
   examples/training/sts/README
   examples/training/nli/README
   examples/training/quora_duplicate_questions/README
   examples/training/multilingual/README



.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   docs/package_reference/models
   docs/package_reference/losses
   docs/package_reference/evaluation
   docs/package_reference/datasets
