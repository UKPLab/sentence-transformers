Quickstart
==========

Sentence Transformer
--------------------

Characteristics of Sentence Transformer (a.k.a bi-encoder) models:

1. Calculates a **fixed-size vector representation (embedding)** given **texts or images**.
2. Embedding calculation is often **efficient**, embedding similarity calculation is **very fast**.
3. Applicable for a **wide range of tasks**, such as semantic textual similarity, semantic search, clustering, classification, paraphrase mining, and more.
4. Often used as a **first step in a two-step retrieval process**, where a Cross-Encoder (a.k.a. reranker) model is used to re-rank the top-k results from the bi-encoder.

Once you have `installed <installation.md>`_ Sentence Transformers, you can easily use Sentence Transformer models:

.. sidebar:: Documentation

   1. :class:`SentenceTransformer <sentence_transformers.SentenceTransformer>`
   2. :meth:`SentenceTransformer.encode <sentence_transformers.SentenceTransformer.encode>`
   3. :meth:`SentenceTransformer.similarity <sentence_transformers.SentenceTransformer.similarity>`

   **Other useful methods and links:**

   - :meth:`SentenceTransformer.similarity_pairwise <sentence_transformers.SentenceTransformer.similarity_pairwise>`
   - `SentenceTransformer > Usage <./sentence_transformer/usage/usage.html>`_
   - `SentenceTransformer > Pretrained Models <./sentence_transformer/pretrained_models.html>`_
   - `SentenceTransformer > Training Overview <./sentence_transformer/training_overview.html>`_
   - `SentenceTransformer > Dataset Overview <./sentence_transformer/dataset_overview.html>`_
   - `SentenceTransformer > Loss Overview <./sentence_transformer/loss_overview.html>`_
   - `SentenceTransformer > Training Examples <./sentence_transformer/training/examples.html>`_

::

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

With ``SentenceTransformer("all-MiniLM-L6-v2")`` we pick which `Sentence Transformer model <https://huggingface.co/models?library=sentence-transformers>`_ we load. In this example, we load `all-MiniLM-L6-v2 <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>`_, which is a MiniLM model finetuned on a large dataset of over 1 billion training pairs. Using `SentenceTransformer.similarity() <./package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.similarity>`_, we compute the similarity between all pairs of sentences. As expected, the similarity between the first two sentences (0.6660) is higher than the similarity between the first and the third sentence (0.1046) or the second and the third sentence (0.1411).

Finetuning Sentence Transformer models is easy and requires only a few lines of code. For more information, see the `Training Overview <./sentence_transformer/training_overview.html>`_ section.

Cross Encoder
-------------

Characteristics of Cross Encoder (a.k.a reranker) models:

1. Calculates a **similarity score** given **pairs of texts**.
2. Generally provides **superior performance** compared to a Sentence Transformer (a.k.a. bi-encoder) model.
3. Often **slower** than a Sentence Transformer model, as it requires computation for each pair rather than each text.
4. Due to the previous 2 characteristics, Cross Encoders are often used to **re-rank the top-k results** from a Sentence Transformer model.

The usage for Cross Encoder (a.k.a. reranker) models is similar to Sentence Transformers:

.. sidebar:: Documentation

   1. :class:`CrossEncoder <sentence_transformers.CrossEncoder>`
   2. :meth:`CrossEncoder.rank <sentence_transformers.CrossEncoder.rank>`
   3. :meth:`CrossEncoder.predict <sentence_transformers.CrossEncoder.predict>`

   **Other useful methods and links:**

   - `CrossEncoder > Usage <./cross_encoder/usage/usage.html>`_
   - `CrossEncoder > Pretrained Models <./cross_encoder/pretrained_models.html>`_
   - `CrossEncoder > Training Overview <./cross_encoder/training_overview.html>`_
   - `CrossEncoder > Dataset Overview <./cross_encoder/dataset_overview.html>`_
   - `CrossEncoder > Loss Overview <./cross_encoder/loss_overview.html>`_
   - `CrossEncoder > Training Examples <./cross_encoder/training/examples.html>`_

::

   from sentence_transformers.cross_encoder import CrossEncoder

   # 1. Load a pretrained CrossEncoder model
   model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

   # We want to compute the similarity between the query sentence...
   query = "A man is eating pasta."

   # ... and all sentences in the corpus
   corpus = [
       "A man is eating food.",
       "A man is eating a piece of bread.",
       "The girl is carrying a baby.",
       "A man is riding a horse.",
       "A woman is playing violin.",
       "Two men pushed carts through the woods.",
       "A man is riding a white horse on an enclosed ground.",
       "A monkey is playing drums.",
       "A cheetah is running behind its prey.",
   ]

   # 2. We rank all sentences in the corpus for the query
   ranks = model.rank(query, corpus)

   # Print the scores
   print("Query: ", query)
   for rank in ranks:
       print(f"{rank['score']:.2f}\t{corpus[rank['corpus_id']]}")
   """
   Query:  A man is eating pasta.
   0.67    A man is eating food.
   0.34    A man is eating a piece of bread.
   0.08    A man is riding a horse.
   0.07    A man is riding a white horse on an enclosed ground.
   0.01    The girl is carrying a baby.
   0.01    Two men pushed carts through the woods.
   0.01    A monkey is playing drums.
   0.01    A woman is playing violin.
   0.01    A cheetah is running behind its prey.
   """

   # 3. Alternatively, you can also manually compute the score between two sentences
   import numpy as np

   sentence_combinations = [[query, sentence] for sentence in corpus]
   scores = model.predict(sentence_combinations)

   # Sort the scores in decreasing order to get the corpus indices
   ranked_indices = np.argsort(scores)[::-1]
   print("Scores:", scores)
   print("Indices:", ranked_indices)
   """
   Scores: [0.6732372, 0.34102544, 0.00542465, 0.07569341, 0.00525378, 0.00536814, 0.06676237, 0.00534825, 0.00516717]
   Indices: [0 1 3 6 2 5 7 4 8]
   """

With ``CrossEncoder("cross-encoder/stsb-distilroberta-base")`` we pick which `CrossEncoder model <./cross_encoder/pretrained_models.html>`_ we load. In this example, we load `cross-encoder/stsb-distilroberta-base <https://huggingface.co/cross-encoder/stsb-distilroberta-base>`_, which is a `DistilRoBERTa <https://huggingface.co/distilbert/distilroberta-base>`_ model finetuned on the `STS Benchmark <https://huggingface.co/datasets/sentence-transformers/stsb>`_ dataset.

Next Steps
----------

Consider reading one of the following sections next:

* `Sentence Transformers > Usage <./sentence_transformer/usage/usage.html>`_
* `Sentence Transformers > Pretrained Models <./sentence_transformer/pretrained_models.html>`_
* `Sentence Transformers > Training Overview <./sentence_transformer/training_overview.html>`_
* `Sentence Transformers > Training Examples > Multilingual Models <../examples/training/multilingual/README.html>`_
* `Cross Encoder > Usage <./cross_encoder/usage/usage.html>`_
* `Cross Encoder > Pretrained Models <./cross_encoder/pretrained_models.html>`_

