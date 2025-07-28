# Loss Overview

```{eval-rst}
.. warning:: 
    To train a :class:`~sentence_transformers.sparse_encoder.SparseEncoder`, you need either :class:`~sentence_transformers.sparse_encoder.losses.SpladeLoss` or :class:`~sentence_transformers.sparse_encoder.losses.CSRLoss`, depending on the architecture. These are wrapper losses that add sparsity regularization on top of a main loss function, which must be provided as a parameter. The only loss that can be used independently is :class:`~sentence_transformers.sparse_encoder.losses.SparseMSELoss`, as it performs embedding-level distillation, ensuring sparsity by directly copying the teacher's sparse embedding.
    
```

## Sparse specific Loss Functions

### SPLADE Loss

The <a href="../package_reference/sparse_encoder/losses.html#spladeloss"><code>SpladeLoss</code></a> implements a specialized loss function for SPLADE (Sparse Lexical and Expansion) models. It combines a main loss function with regularization terms to control efficiency:

- Supports all the losses mention below as main loss but three principal loss types: <a href="../package_reference/sparse_encoder/losses.html#sparsemultiplenegativesrankingloss"><code>SparseMultipleNegativesRankingLoss</code></a>, <a href="../package_reference/sparse_encoder/losses.html#sparsemarginmseloss"><code>SparseMarginMSELoss</code></a> and <a href="../package_reference/sparse_encoder/losses.html#sparsedistilkldivloss"><code>SparseDistillKLDivLoss</code></a>.
- Uses <a href="../package_reference/sparse_encoder/losses.html#flopsloss"><code>FlopsLoss</code></a> for regularization to control sparsity by default, but supports custom regularizers.
- Balances effectiveness (via the main loss) with efficiency by regularizing both query and document representations.
- Allows using different regularizers for queries and documents via the `query_regularizer` and `document_regularizer` parameters, enabling fine-grained control over sparsity patterns for different types of inputs.
- Supports separate threshold values for queries and documents via the `query_regularizer_threshold` and `document_regularizer_threshold` parameters, allowing different sparsity strictness levels for each input type.

### CSR Loss

If you are using the <a href="../package_reference/sparse_encoder/models.html#sparseautoencoder"><code>SparseAutoEncoder</code></a> module, then you have to use the <a href="../package_reference/sparse_encoder/losses.html#csrloss"><code>CSRLoss</code></a> (Contrastive Sparse Representation Loss). It combines two components:

- A reconstruction loss <a href="../package_reference/sparse_encoder/losses.html#csrreconstructionloss"><code>CSRReconstructionLoss</code></a> that ensures sparse representation can faithfully reconstruct original embeddings.
- A main loss, which in the paper is a contrastive learning component using <a href="../package_reference/sparse_encoder/losses.html#sparsemultiplenegativesrankingloss">`SparseMultipleNegativesRankingLoss`</a> that ensures semanticallysimilar sentences have similar representations. But it's theorically possible to use all the losses mention below as main loss like for <a href="../package_reference/sparse_encoder/losses.html#spladeloss"><code>SpladeLoss</code></a> .


## Loss Table

Loss functions play a critical role in the performance of your fine-tuned model. Sadly, there is no "one size fits all" loss function. Ideally, this table should help narrow down your choice of loss function(s) by matching them to your data formats.

```{eval-rst}
.. note:: 

    You can often convert one training data format into another, allowing more loss functions to be viable for your scenario. For example, ``(sentence_A, sentence_B) pairs`` with ``class`` labels can be converted into ``(anchor, positive, negative) triplets`` by sampling sentences with the same or different classes.
 
 .. note:: 

    The loss functions in `SentenceTransformer > Loss Overview <../sentence_transformer/loss_overview.html>`_ that appear here with the ``Sparse`` prefix are identical to their dense versions. The prefix is used only to indicate which losses can be used as main losses to train a :class:`~sentence_transformers.sparse_encoder.SparseEncoder`
```

| Inputs                                            | Labels                                   | Appropriate Loss Functions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|---------------------------------------------------|------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `(anchor, positive) pairs`                        | `none`                                   | <a href="../package_reference/sparse_encoder/losses.html#sparsemultiplenegativesrankingloss">`SparseMultipleNegativesRankingLoss`</a>                      |
| `(sentence_A, sentence_B) pairs`                  | `float similarity score between 0 and 1` | <a href="../package_reference/sparse_encoder/losses.html#sparsecosentloss">`SparseCoSENTLoss`</a><br><a href="../package_reference/sparse_encoder/losses.html#sparseangleloss">`SparseAnglELoss`</a><br><a href="../package_reference/sparse_encoder/losses.html#sparsecosinesimilarityloss">`SparseCosineSimilarityLoss`</a>                                                                                                                                                                                                                                                                                                       |
| `(anchor, positive, negative) triplets`           | `none`                                   | <a href="../package_reference/sparse_encoder/losses.html#sparsemultiplenegativesrankingloss">`SparseMultipleNegativesRankingLoss`</a><br><a href="../package_reference/sparse_encoder/losses.html#sparsetripletloss">`SparseTripletLoss`</a> |
| `(anchor, positive, negative_1, ..., negative_n)` | `none`                                   | <a href="../package_reference/sparse_encoder/losses.html#sparsemultiplenegativesrankingloss">`SparseMultipleNegativesRankingLoss`</a>                                                                                                                                    |


## Distillation
These loss functions are specifically designed to be used when distilling the knowledge from one model into another. This is rather commonly used when training Sparse embedding models.

| Texts                                             | Labels                                                                    | Appropriate Loss Functions                                                                                                                                                                                              |
|---------------------------------------------------|---------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `sentence`                                        | `model sentence embeddings`                                               | <a href="../package_reference/sparse_encoder/losses.html#sparsemseloss">`SparseMSELoss`</a>                                                                                                                             |
| `sentence_1, sentence_2, ..., sentence_N`         | `model sentence embeddings`                                               | <a href="../package_reference/sparse_encoder/losses.html#sparsemseloss">`SparseMSELoss`</a>                                                                                                                             |
| `(query, passage_one, passage_two) triplets`      | `gold_sim(query, passage_one) - gold_sim(query, passage_two)`             | <a href="../package_reference/sparse_encoder/losses.html#sparsemarginmseloss">`SparseMarginMSELoss`</a>                                                                                                                 |
| `(query, positive, negative) triplets`            | `[gold_sim(query, positive), gold_sim(query, negative)]`                  | <a href="../package_reference/sparse_encoder/losses.html#sparsedistilkldivloss">`SparseDistillKLDivLoss`</a><br><a href="../package_reference/sparse_encoder/losses.html#sparsemarginmseloss">`SparseMarginMSELoss`</a> |
| `(query, positive, negative_1, ..., negative_n)`  | `[gold_sim(query, positive) - gold_sim(query, negative_i) for i in 1..n]` | <a href="../package_reference/sparse_encoder/losses.html#sparsemarginmseloss">`SparseMarginMSELoss`</a>                                                                                                                 |
| `(query, positive, negative_1, ..., negative_n) ` | `[gold_sim(query, positive), gold_sim(query, negative_i)...] `            | <a href="../package_reference/sparse_encoder/losses.html#sparsedistilkldivloss">`SparseDistillKLDivLoss`</a><br><a href="../package_reference/sparse_encoder/losses.html#sparsemarginmseloss">`SparseMarginMSELoss`</a> |


## Commonly used Loss Functions

In practice, not all loss functions get used equally often. The most common scenarios are:

* `(anchor, positive) pairs` without any labels: <a href="../package_reference/sparse_encoder/losses.html#sparsemultiplenegativesrankingloss"><code>SparseMultipleNegativesRankingLoss</code></a> (a.k.a. InfoNCE or in-batch negatives loss) is commonly used to train the top performing embedding models. This data is often relatively cheap to obtain, and the models are generally very performant. Here for our sparse retrieval tasks, this format works well with <a href="../package_reference/sparse_encoder/losses.html#spladeloss"><code>SpladeLoss</code></a> or <a href="../package_reference/sparse_encoder/losses.html#csrloss"><code>CSRLoss</code></a>, both typically using InfoNCE as their underlying loss function.

* `(query, positive, negative_1, ..., negative_n)` format: This structure with multiple negatives is particularly effective with <a href="../package_reference/sparse_encoder/losses.html#spladeloss"><code>SpladeLoss</code></a> configured with <a href="../package_reference/sparse_encoder/losses.html#sparsemarginmseloss"><code>SparseMarginMSELoss</code></a>, especially in knowledge distillation scenarios where a teacher model provides similarity scores. The strongest models are trained with distillation losses like <a href="../package_reference/sparse_encoder/losses.html#sparsedistilkldivloss"><code>SparseDistillKLDivLoss</code></a> or <a href="../package_reference/sparse_encoder/losses.html#sparsemarginmseloss"><code>SparseMarginMSELoss</code></a>.

## Custom Loss Functions

```{eval-rst}
Advanced users can create and train with their own loss functions. Custom loss functions only have a few requirements:

- They must be a subclass of :class:`torch.nn.Module`.
- They must have ``model`` as the first argument in the constructor.
- They must implement a ``forward`` method that accepts ``sentence_features`` and ``labels``. The former is a list of tokenized batches, one element for each column. These tokenized batches can be fed directly to the ``model`` being trained to produce embeddings. The latter is an optional tensor of labels. The method must return a single loss value or a dictionary of loss components (component names to loss values) that will be summed to produce the final loss value. When returning a dictionary, the individual components will be logged separately in addition to the summed loss, allowing you to monitor the individual components of the loss.

To get full support with the automatic model card generation, you may also wish to implement:

- a ``get_config_dict`` method that returns a dictionary of loss parameters.
- a ``citation`` property so your work gets cited in all models that train with the loss.

Consider inspecting existing loss functions to get a feel for how loss functions are commonly implemented.
```