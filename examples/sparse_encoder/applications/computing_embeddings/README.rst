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

Controlling Sparsity with max_active_dims
-----------------------------------------

For sparse models, you can control the maximum number of active dimensions (non-zero values) in the output embeddings using the ``max_active_dims`` parameter. This is particularly useful for:

1. Reducing memory usage and storage requirements
2. Improving search efficiency in production systems
3. Controlling the trade-off between accuracy and performance

You can specify ``max_active_dims`` either when initializing the model or during encoding:

::

   from sentence_transformers import SparseEncoder

   # Initialize the SPLADE model
   model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

   # Embed a list of sentences
   sentences = [
      "This framework generates embeddings for each input sentence",
      "Sentences are passed as a list of string.",
      "The quick brown fox jumps over the lazy dog.",
   ]

   # Generate embeddings
   embeddings = model.encode(sentences)

   # Print embedding sim and sparsity
   print(f"Embedding dim: {model.get_sentence_embedding_dimension()}")

   stats = model.get_sparsity_stats(embeddings)
   print(f"Embedding sparsity: {stats}")
   print(f"Average non-zero dimensions: {stats['row_non_zero_mean']:.2f}")
   print(f"Sparsity percentage: {stats['row_sparsity_mean']:.2%}")


   """
   Embedding dim: 30522
   Embedding sparsity: {'num_rows': 3, 'num_cols': 30522, 'row_non_zero_mean': 56.66666793823242, 'row_sparsity_mean': 0.9981433749198914}
   Average non-zero dimensions: 56.67
   Sparsity percentage: 99.81%
   """
   
   # Example of using max_active_dims during encoding
   print("\n--- Using max_active_dims during encoding ---")
   # Generate embeddings with limited active dimensions
   embeddings_limited = model.encode(sentences, max_active_dims=32)
   stats_limited = model.get_sparsity_stats(embeddings_limited)
   print(f"Limited embedding sparsity: {stats_limited}")
   print(f"Average non-zero dimensions: {stats_limited['row_non_zero_mean']:.2f}")
   print(f"Sparsity percentage: {stats_limited['row_sparsity_mean']:.2%}")

   """
   --- Using max_active_dims during encoding ---
   Limited embedding sparsity: {'num_rows': 3, 'num_cols': 30522, 'row_non_zero_mean': 32.0, 'row_sparsity_mean': 0.9989516139030457}
   Average non-zero dimensions: 32.00
   Sparsity percentage: 99.90%
   """

When you set ``max_active_dims``, the model will keep only the top-K dimensions with the highest values and set all other values to zero. This ensures your embeddings maintain a controlled level of sparsity while preserving the most important semantic information.

.. note::

   Setting a very low ``max_active_dims`` value may reduce the quality of search results. The optimal value depends on your specific use case and dataset. Common values range from 128 to 512 dimensions, like dense embeddings dimensions.


One of the key benefits of controlling sparsity with ``max_active_dims`` is reduced memory usage. Here's an example showing the memory savings:

::

   # Compare memory usage between default and limited active dimensions
   print("\n--- Comparing memory usage ---")
   def get_memory_size(tensor):
       if tensor.is_sparse:
           # For sparse tensors, only count non-zero elements
           return (tensor._values().element_size() * tensor._values().nelement() + 
                  tensor._indices().element_size() * tensor._indices().nelement())
       else:
           return tensor.element_size() * tensor.nelement()

   print(f"Original embeddings memory: {get_memory_size(embeddings) / 1024:.2f} KB")
   print(f"Embeddings with max_active_dims=32 memory: {get_memory_size(embeddings_limited) / 1024:.2f} KB")

   """
   --- Comparing memory usage ---
   Original embeddings memory: 3.32 KB
   Embeddings with max_active_dims=32 memory: 1.88 KB
   """

As shown in the example, limiting active dimensions to 32 reduced memory usage by approximately 43%. This efficiency becomes even more significant when working with large document collections but need to be put in balance with the possible loss of quality of the embeddings representations.

Interpretability with SPLADE Models
----------------------------------

When using SPLADE models, a key advantage is interpretability. You can easily visualize which tokens contribute most to the embedding, providing insights into what the model considers important in the text:

::

   from sentence_transformers import SparseEncoder

   # Initialize the SPLADE model
   model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

   # Embed a list of sentences
   sentences = [
      "This framework generates embeddings for each input sentence",
      "Sentences are passed as a list of string.",
      "The quick brown fox jumps over the lazy dog.",
   ]

   # Generate embeddings
   embeddings = model.encode(sentences)

   # Visualize top tokens for each text
   top_k = 10

   token_weights = model.decode(embeddings, top_k=top_k)

   print(f"\nTop tokens {top_k} for each text:")
   # The result is a list of sentence embeddings as numpy arrays
   for i, sentence in enumerate(sentences):
      token_scores = ", ".join([f'("{token.strip()}", {value:.2f})' for token, value in token_weights[i]])
      print(f"{i}: {sentence} -> Top tokens:  {token_scores}")

   """
   Top tokens 10 for each text:
      0: This framework generates embeddings for each input sentence -> Top tokens:  ("framework", 2.19), ("##bed", 2.12), ("input", 1.99), ("each", 1.60), ("em", 1.58), ("sentence", 1.49), ("generate", 1.42), ("##ding", 1.33), ("sentences", 1.10), ("create", 0.93)
      1: Sentences are passed as a list of string. -> Top tokens:  ("string", 2.72), ("pass", 2.24), ("sentences", 2.15), ("passed", 2.07), ("sentence", 1.90), ("strings", 1.86), ("list", 1.84), ("lists", 1.49), ("as", 1.18), ("passing", 0.73)
      2: The quick brown fox jumps over the lazy dog. -> Top tokens:  ("lazy", 2.18), ("fox", 1.67), ("brown", 1.56), ("over", 1.52), ("dog", 1.50), ("quick", 1.49), ("jump", 1.39), ("dogs", 1.25), ("foxes", 0.99), ("jumping", 0.84)
   """
   
This interpretability helps in understanding why certain documents match or don't match in search applications, and provides transparency into the model's behavior.
