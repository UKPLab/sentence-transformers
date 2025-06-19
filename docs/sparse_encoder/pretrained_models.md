# Pretrained SPLADE Models

Currently, no base models are available from our organization but the community has provided these powerful models for sparse neural retrieval.

Models integrate seamlessly with this simple interface:

```python
from sentence_transformers import SparseEncoder

# Download from the 🤗 Hub
model = SparseEncoder("naver/splade-v3")
# Run inference
queries = ["what causes aging fast"]
documents = [
    "UV-A light, specifically, is what mainly causes tanning, skin aging, and cataracts, UV-B causes sunburn, skin aging and skin cancer, and UV-C is the strongest, and therefore most effective at killing microorganisms. Again â\x80\x93 single words and multiple bullets.",
    "Answers from Ronald Petersen, M.D. Yes, Alzheimer's disease usually worsens slowly. But its speed of progression varies, depending on a person's genetic makeup, environmental factors, age at diagnosis and other medical conditions. Still, anyone diagnosed with Alzheimer's whose symptoms seem to be progressing quickly â\x80\x94 or who experiences a sudden decline â\x80\x94 should see his or her doctor.",
    "Bell's palsy and Extreme tiredness and Extreme fatigue (2 causes) Bell's palsy and Extreme tiredness and Hepatitis (2 causes) Bell's palsy and Extreme tiredness and Liver pain (2 causes) Bell's palsy and Extreme tiredness and Lymph node swelling in children (2 causes)",
]
query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# [1, 30522] [3, 30522]

# Get the similarity scores for the embeddings
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
# tensor([[11.3768, 10.8296,  4.3457]])
```

[MS MARCO Passage Retrieval](https://github.com/microsoft/MSMARCO-Passage-Ranking) serves as the gold standard dataset, featuring authentic user queries from Bing search engine paired with expertly annotated relevant text passages. Models trained on this benchmark demonstrate exceptional effectiveness as embedding models for production search systems. Performance scores reflect evaluation on this dataset, it's a good indication but shouldn't be the only paramters to take into account.

[BEIR (Benchmarking IR)](https://github.com/beir-cellar/beir) provides a heterogeneous benchmark for evaluation of information retrieval models across in our case 13 diverse datasets. The avg nDCG@10 scores represent the average performance across all 13 datasets.

Note that all the numbers of below are extracted information from differents papers.
## Core SPLADE Models

These models represent the backbone of sparse neural retrieval:

| Model Name | MRR@10 | avg nDCG@10 (BEIR-13) |
|------------|:------:|:---------------------:|
| [opensearch-project/opensearch-neural-sparse-encoding-v2-distill](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-v2-distill) | NA | **52.8** |
| [opensearch-project/opensearch-neural-sparse-encoding-v1](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-v1) | NA | 52.4 |
| [naver/splade-v3](https://huggingface.co/naver/splade-v3) | **40.2** | 51.7 |
| [ibm-granite/granite-embedding-30m-sparse](https://huggingface.co/ibm-granite/granite-embedding-30m-sparse) | NA | 50.8 |
| [naver/splade-cocondenser-selfdistil](https://huggingface.co/naver/splade-cocondenser-selfdistil) | 37.6 | 50.7 |
| [naver/splade_v2_distil](https://huggingface.co/naver/splade_v2_distil) | 36.8 | 50.6 |
| [naver/splade-cocondenser-ensembledistil](https://huggingface.co/naver/splade-cocondenser-ensembledistil) | 38.0 | 50.5 |
| [naver/splade-v3-distilbert](https://huggingface.co/naver/splade-v3-distilbert) | 38.7 | 50.0 |
| [prithivida/Splade_PP_en_v2](https://huggingface.co/prithivida/Splade_PP_en_v2) | 37.8 | 49.4 
| [naver/splade-v3-lexical](https://huggingface.co/naver/splade-v3-lexical) | 40.0 | 49.1 |
| [prithivida/Splade_PP_en_v1](https://huggingface.co/prithivida/Splade_PP_en_v1) | 37.2| 48.7 |
| [naver/splade_v2_max](https://huggingface.co/naver/splade_v2_max) | 34.0 | 46.4 |

## Inference-Free SPLADE Models

```{eval-rst}
Inference-free Splade uses for the documents part a traditional Splade architecture and for the query part is an :class:`~sentence_transformers.sparse_encoder.models.IDF` module, which just returns a pre-computed score for every token in the query. So for this models we loose the query-expension but we have now no models inference at the query time which is very valuable for speed optimization.
```

| Model Name | avg nDCG@10 (BEIR-13) |
|------------|:---------------------:|
| [opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte) | **54.6** |
| [opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill) | 51.7 |
| [opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill) | 50.4 |
| [opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini) | 49.7 |
| [opensearch-project/opensearch-neural-sparse-encoding-doc-v1](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v1) | 49.0 |
| [naver/splade-v3-doc](https://huggingface.co/naver/splade-v3-doc) | 47.0 |

## Essential Resources

- [**SPLADE Models Collection**](https://huggingface.co/collections/arthurbresnu/splade-models-6850b6f81b29d17efc0171de) - Comprehensive model repository on Hugging Face
- [**Inference-Free SPLADE Models**](https://huggingface.co/collections/arthurbresnu/inference-free-splade-models-6850b79e9becbb2eb0bbbd28) - Optimized deployment-ready models
