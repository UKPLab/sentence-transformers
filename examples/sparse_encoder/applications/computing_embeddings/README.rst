Computing Sparse Embeddings
===========================

Once you have `installed <../../../../docs/installation.html>`_ Sentence Transformers, you can easily use Sparse Encoder models:

.. sidebar:: Documentation

   1. :class:`SparseEncoder <sentence_transformers.sparse_encoder.SparseEncoder>`
   2. :meth:`SparseEncoder.encode <sentence_transformers.sparse_encoder.SparseEncoder.encode>`
   3. :meth:`SparseEncoder.similarity <sentence_transformers.sparse_encoder.SparseEncoder.similarity>`

::

   from sentence_transformers import SparseEncoder

   # 1. Load a pretrained SparseEncoder model
   model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

   # The sentences to encode
   sentences = [
       "The weather is lovely today.",
       "It's so sunny outside!",
       "He drove to the stadium.",
   ]

   # 2. Calculate sparse embeddings by calling model.encode()
   embeddings = model.encode(sentences)
   print(embeddings.shape)
   # [3, 30522] - sparse representation with vocabulary size dimensions

   # 3. Calculate the embedding similarities (using dot product by default)
   similarities = model.similarity(embeddings, embeddings)
   print(similarities)
   # tensor([[   35.6293,     9.1541,     0.1127],
   #         [    9.1541,    27.4778,     0.0191],
   #         [    0.1127,     0.0191,    29.6122]], device='cuda:0')

   # 4. Check sparsity statistics
   stats = SparseEncoder.get_sparsity_stats(embeddings)
   print(f"Sparsity: {stats['row_sparsity_mean']:.2%}")  # Typically >99% zeros
   print(f"Avg non-zero dimensions per embedding: {stats['row_non_zero_mean']:.2f}")

.. note::
   Even though we talk about sentence embeddings, you can use Sparse Encoder for shorter phrases as well as for longer texts with multiple sentences. See :ref:`input-sequence-length` for notes on embeddings for longer texts.


Initializing a Sparse Encoder Model
-----------------------------------

The first step is to load a pretrained Sparse Encoder model. You can use any of the models from the `Pretrained Models <../../../../docs/sparse_encoder/pretrained_models.html>`_ or a local model. See also :class:`~sentence_transformers.sparse_encoder.SparseEncoder` for information on parameters.

::

   from sentence_transformers import SparseEncoder

   model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
   # Alternatively, you can pass a path to a local model directory:
   model = SparseEncoder("output/models/sparse-distilbert-nq-finetuned")

The model will automatically be placed on the most performant available device, e.g. ``cuda`` or ``mps`` if available. You can also specify the device explicitly:

::

   model = SparseEncoder("naver/splade-cocondenser-ensembledistil", device="cuda")

Calculating Embeddings
----------------------

The method to calculate embeddings is :meth:`SparseEncoder.encode <sentence_transformers.sparse_encoder.SparseEncoder.encode>`.

.. _input-sequence-length:
Input Sequence Length
---------------------

For transformer models like BERT, RoBERTa, DistilBERT etc., the runtime and memory requirement grows quadratic with the input length. This limits transformers to inputs of certain lengths. A common value for BERT-based models are 512 tokens, which corresponds to about 300-400 words (for English).

Each model has a maximum sequence length under ``model.max_seq_length``, which is the maximal number of tokens that can be processed. Longer texts will be truncated to the first ``model.max_seq_length`` tokens::

    from sentence_transformers import SparseEncoder

    model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
    print("Max Sequence Length:", model.max_seq_length)
    # => Max Sequence Length: 256

    # Change the length to 200
    model.max_seq_length = 200

    print("Max Sequence Length:", model.max_seq_length)
    # => Max Sequence Length: 200

.. note::

   You cannot increase the length higher than what is maximally supported by the respective transformer model. Also note that if a model was trained on short texts, the representations for long texts might not be that good.
