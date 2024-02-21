# Matryoshka Embeddings

Dense embedding models typically produce embeddings with a fixed size, such as 768 or 1024. All further computations (clustering, classification, semantic search, retrieval, reranking, etc.) must then be done on these full embeddings. [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) revisits this idea, and proposes a solution to train embedding models whose embeddings are still useful after truncation to much smaller sizes. This allows for considerably faster (bulk) processing.

## Use Cases

A particularly interesting use case is to split up processing into two steps: 1) pre-processing with much smaller vectors and then 2) processing the remaining vectors as full size (also called "shortlisting and reranking"). Additionally, Matryoshka models will allow you to scale your embedding solutions to your desired storage cost, processing speed and performance.

## Training

Training using Matryoshka Representation Learning (MRL) is quite elementary: rather than applying some loss function on only the full-size embeddings, we also apply that same loss function on truncated portions of the embeddings. For example, if a model has an embedding dimension of 768 by default, it can now be trained on 768, 512, 256, 128, 64 and 32. Each of these losses will be added together, optionally with some weight:

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CoSENTLoss, MatryoshkaLoss

model = SentenceTransformer("microsoft/mpnet-base")

base_loss = CoSENTLoss(model=model)
loss = MatryoshkaLoss(model=model, loss=base_loss, matryoshka_dims=[768, 512, 256, 128, 64])
```
* **Reference**: [`MatryoshkaLoss`](../../../docs/package_reference/losses#matryoshkaloss)

## Inference

After a model has been trained using a Matryoshka loss, you can then run inference with it using <a href="../../../docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode"><code>SentenceTransformers.encode</code></a>. You must then truncate the resulting embeddings, and it is recommended to renormalize the embeddings.

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

matryoshka_dim = 64
embeddings = model.encode(
    [
        "search_query: What is TSNE?",
        "search_document: t-distributed stochastic neighbor embedding (t-SNE) is a statistical method for visualizing high-dimensional data by giving each datapoint a location in a two or three-dimensional map.",
        "search_document: Amelia Mary Earhart was an American aviation pioneer and writer.",
    ]
)
embeddings[..., :matryoshka_dim]  # Shrink the embedding dimensions

similarities = cos_sim(embeddings[0], embeddings[1:])
# => tensor([[0.7839, 0.4933]])
```
As you can see, the similarity between the search query and the correct document is much higher than that of an unrelated document, despite the very small matryoshka dimension applied. Feel free to copy this script locally, modify the `matryoshka_dim`, and observe the difference in similarities.

**Note**: Despite the embeddings being smaller, training and inference of a Matryoshka model is not faster, not more memory-efficient, and not smaller. Only the processing and storage of the resulting embeddings will be faster and cheaper.

## Code Examples

See the following scripts as examples of how to apply the <a href="../../../docs/package_reference/losses.html#matryoshkaloss"><code>MatryoshkaLoss</code></a> in practice:

* **[matryoshka_nli.py](matryoshka_nli.py)**: This example uses the MultipleNegativesRankingLoss with MatryoshkaLoss to train a strong embedding model using Natural Language Inference (NLI) data. It is an adaptation of the [NLI](../nli/README) documentation.
* **[matryoshka_nli_reduced_dim.py](matryoshka_nli_reduced_dim.py)**: This example uses the MultipleNegativesRankingLoss with MatryoshkaLoss to train a strong embedding model with a small maximum output dimension of 256. It trains using Natural Language Inference (NLI) data, and is an adaptation of the [NLI](../nli/README) documentation.
* **[matryoshka_sts.py](matryoshka_sts.py)**: This example uses the CoSENTLoss with MatryoshkaLoss to train an embedding model on the training set of the STSBenchmark dataset. It is an adaptation of the [STS](../sts/README) documentation.
