
Usage
=====

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

.. toctree::
   :maxdepth: 1
   :caption: Tasks and Advanced Usage

   ../../../examples/applications/computing-embeddings/README
   semantic_textual_similarity
   ../../../examples/applications/semantic-search/README
   ../../../examples/applications/retrieve_rerank/README
   ../../../examples/applications/clustering/README
   ../../../examples/applications/paraphrase-mining/README
   ../../../examples/applications/parallel-sentence-mining/README
   ../../../examples/applications/image-search/README
   ../../../examples/applications/embedding-quantization/README

