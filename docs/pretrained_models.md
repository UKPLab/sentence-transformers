# Pretrained Models

We provide various pre-trained models. Using these models is easy:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("model_name")
```

All models are hosted on the [HuggingFace Model Hub](https://huggingface.co/sentence-transformers).

## Model Overview

The following table provides an overview of (selected) models. They have been extensively evaluated for their quality to embedded sentences (Performance Sentence Embeddings) and to embedded search queries & paragraphs (Performance Semantic Search).

The **all-*** models were trained on all available training data (more than 1 billion training pairs) and are designed as **general purpose** models. The **all-mpnet-base-v2** model provides the best quality, while **all-MiniLM-L6-v2** is 5 times faster and still offers good quality. Toggle *All models* to see all evaluated models or visit [HuggingFace Model Hub](https://huggingface.co/models?library=sentence-transformers) to view all existing sentence-transformers models. 


<iframe src="../_static/html/models_en_sentence_embeddings.html" height="600" style="width:100%; border:none;" title="Iframe Example"></iframe>

---

## Semantic Search

The following models have been specifically trained for **Semantic Search**: Given a question / search query, these models are able to find relevant text passages. For more details, see [Usage - Semantic Search](../examples/applications/semantic-search/README.md).

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

query_embedding = model.encode("How big is London")
passage_embedding = model.encode([
    "London has 9,787,426 inhabitants at the 2011 census",
    "London is known for its finacial district",
])

print("Similarity:", util.dot_score(query_embedding, passage_embedding))
```



### Multi-QA Models

The following models have been trained on [215M question-answer pairs](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-dot-v1#training) from various sources and domains, including StackExchange, Yahoo Answers, Google & Bing search queries and many more. These model perform well across many search tasks and domains.


These models were tuned to be used with dot-product:

| Model | Performance Semantic Search (6 Datasets) | Queries (GPU / CPU) per sec. | 
| --- | :---: | :---: |
| [multi-qa-MiniLM-L6-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-dot-v1) | 49.19 | 18,000 / 750 |
| [multi-qa-distilbert-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-distilbert-dot-v1) | 52.51  | 7,000 / 350 |
| [multi-qa-mpnet-base-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1) | 57.60 | 4,000 / 170 |



These models produce normalized vectors of length 1, which can be used with dot-product, cosine-similarity and Euclidean distance:

| Model | Performance Semantic Search (6 Datasets) | Queries (GPU / CPU) per sec. | 
| --- | :---: | :---: |
| [multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1) | 51.83 | 18,000 / 750 |
| [multi-qa-distilbert-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-distilbert-cos-v1) |  52.83 | 7,000 / 350 |
| [multi-qa-mpnet-base-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-cos-v1) | 57.46 | 4,000 / 170 |

### MSMARCO Passage Models

The [MSMARCO Passage Ranking Dataset](https://github.com/microsoft/MSMARCO-Passage-Ranking) contains 500k real queries from Bing search together with the relevant passages from various web sources. Given the diversity of the MSMARCO dataset, models also perform well on other domains. 

Models tuned to be used with dot-product:

| Model | MSMARCO MRR@10 dev set | Performance Semantic Search (6 Datasets) | Queries (GPU / CPU) per sec. | 
| --- | :---: | :---: | :---: |
| [msmarco-distilbert-base-tas-b](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b) | 34.43 | 49.25 | 7,000 / 350 |
| [msmarco-distilbert-dot-v5](https://huggingface.co/sentence-transformers/msmarco-distilbert-dot-v5) | 37.25 | 49.47 | 7,000 / 350 |
| [msmarco-bert-base-dot-v5](https://huggingface.co/sentence-transformers/msmarco-bert-base-dot-v5) | 38.08 | 52.11 | 4,000 / 170 |


These models produce normalized vectors of length 1, which can be used with dot-product, cosine-similarity and Euclidean distance:

| Model | MSMARCO MRR@10 dev set | Performance Semantic Search (6 Datasets) | Queries (GPU / CPU) per sec. | 
| --- | :---: | :---: | :---: |
| [msmarco-MiniLM-L6-cos-v5](https://huggingface.co/sentence-transformers/msmarco-MiniLM-L6-cos-v5) | 32.27 | 42.16 | 18,000 / 750 |
| [msmarco-MiniLM-L12-cos-v5](https://huggingface.co/sentence-transformers/msmarco-MiniLM-L12-cos-v5) | 32.75 | 43.89 | 11,000 / 400 |
| [msmarco-distilbert-cos-v5](https://huggingface.co/sentence-transformers/msmarco-distilbert-cos-v5) | 33.79 | 44.98 | 7,000 / 350 |

[MSMARCO Models - More details](pretrained-models/msmarco-v5.md)

---

## Multi-Lingual Models
The following models generate aligned vector spaces, i.e., similar inputs in different languages are mapped close in vector space. You do not need to specify the input language.  Details are in our publication [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813). We used the following 50+ languages: ar, bg, ca, cs, da, de, el, en, es, et, fa, fi, fr, fr-ca, gl, gu, he, hi, hr, hu, hy, id, it, ja, ka, ko, ku, lt, lv, mk, mn, mr, ms, my, nb, nl, pl, pt, pt-br, ro, ru, sk, sl, sq, sr, sv, th, tr, uk, ur, vi, zh-cn, zh-tw. 



**Semantic Similarity**

These models find semantically similar sentences within one language or across languages:

- **distiluse-base-multilingual-cased-v1**: Multilingual knowledge distilled version of [multilingual Universal Sentence Encoder](https://arxiv.org/abs/1907.04307). Supports 15 languages:  Arabic, Chinese, Dutch, English, French, German, Italian, Korean, Polish, Portuguese, Russian, Spanish, Turkish. 
- **distiluse-base-multilingual-cased-v2**: Multilingual knowledge distilled version of [multilingual Universal Sentence Encoder](https://arxiv.org/abs/1907.04307). This version supports 50+ languages, but performs a bit weaker than the v1 model.
- **paraphrase-multilingual-MiniLM-L12-v2** - Multilingual version of *paraphrase-MiniLM-L12-v2*, trained on parallel data for 50+ languages. 
- **paraphrase-multilingual-mpnet-base-v2** - Multilingual version of *paraphrase-mpnet-base-v2*, trained on parallel data for 50+ languages. 

**Bitext Mining** 

Bitext mining describes the process of finding translated sentence pairs in two languages. If this is your use-case, the following model gives the best performance:
- **LaBSE** - [LaBSE](https://arxiv.org/abs/2007.01852) Model. Supports 109 languages. Works well for finding translation pairs in multiple languages. As detailed  [here](https://arxiv.org/abs/2004.09813), LaBSE works less well for assessing the similarity of sentence pairs that are not translations of each other.


Extending a model to new languages is easy by following [the description here](https://www.sbert.net/examples/training/multilingual/README.html).

----

## Image & Text-Models
The following models can embed images and text into a joint vector space. See [Image Search](../examples/applications/image-search/README.md)  for more details how to use for text2image-search, image2image-search, image clustering, and zero-shot image classification.

The following models are available with their respective Top 1 accuracy on zero-shot ImageNet validation dataset.

| Model | Top 1 Performance |
| --- | :---: |
| [clip-ViT-B-32](https://huggingface.co/sentence-transformers/clip-ViT-B-32) | 63.3 |
| [clip-ViT-B-16](https://huggingface.co/sentence-transformers/clip-ViT-B-16) | 68.1 |
| [clip-ViT-L-14](https://huggingface.co/sentence-transformers/clip-ViT-L-14) | 75.4 |

We further provide this multilingual text-image model:
- **clip-ViT-B-32-multilingual-v1** - Multilingual text encoder for the [clip-ViT-B-32](https://huggingface.co/sentence-transformers/clip-ViT-B-32)   model using [Multilingual Knowledge Distillation](https://arxiv.org/abs/2004.09813). This model can encode text in 50+ languages to match the image vectors from the [clip-ViT-B-32](https://huggingface.co/sentence-transformers/clip-ViT-B-32)  model.


---

## Other Models

### INSTRUCTOR models
As of Sentence Transformers 2.4.0, some INSTRUCTOR models such as [hkunlp/instructor-large](https://huggingface.co/hkunlp/instructor-large) are natively supported in Sentence Transformers. These models are special, as they are trained with instructions in mind. Notably, the primary difference between normal Sentence Transformer models and Instructor models is that the latter do not include the instructions themselves in the pooling step.

The following models work out of the box:
* [hkunlp/instructor-base](https://huggingface.co/hkunlp/instructor-base)
* [hkunlp/instructor-large](https://huggingface.co/hkunlp/instructor-large)
* [hkunlp/instructor-xl](https://huggingface.co/hkunlp/instructor-xl)

You can use these models like so:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("hkunlp/instructor-large")
embeddings = model.encode(
    [
        "Dynamical Scalar Degree of Freedom in Horava-Lifshitz Gravity",
        "Comparison of Atmospheric Neutrino Flux Calculations at Low Energies",
        "Fermion Bags in the Massive Gross-Neveu Model",
        "QCD corrections to Associated t-tbar-H production at the Tevatron",
    ],
    prompt="Represent the Medicine sentence for clustering: ",
)
print(embeddings.shape)
# => (4, 768)
```

For example, for information retrieval:
```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("hkunlp/instructor-large")
query = "where is the food stored in a yam plant"
query_instruction = (
    "Represent the Wikipedia question for retrieving supporting documents: "
)
corpus = [
    'Yams are perennial herbaceous vines native to Africa, Asia, and the Americas and cultivated for the consumption of their starchy tubers in many temperate and tropical regions. The tubers themselves, also called "yams", come in a variety of forms owing to numerous cultivars and related species.',
    "The disparate impact theory is especially controversial under the Fair Housing Act because the Act regulates many activities relating to housing, insurance, and mortgage loansâ€”and some scholars have argued that the theory's use under the Fair Housing Act, combined with extensions of the Community Reinvestment Act, contributed to rise of sub-prime lending and the crash of the U.S. housing market and ensuing global economic recession",
    "Disparate impact in United States labor law refers to practices in employment, housing, and other areas that adversely affect one group of people of a protected characteristic more than another, even though rules applied by employers or landlords are formally neutral. Although the protected classes vary by statute, most federal civil rights laws protect based on race, color, religion, national origin, and sex as protected traits, and some laws include disability status and other traits as well.",
]
corpus_instruction = "Represent the Wikipedia document for retrieval: "

query_embedding = model.encode(query, prompt=query_instruction)
corpus_embeddings = model.encode(corpus, prompt=corpus_instruction)
similarities = cos_sim(query_embedding, corpus_embeddings)
print(similarities)
# => tensor([[0.8835, 0.7037, 0.6970]])
```

All other Instructor models either 1) will not load as they refer to `InstructorEmbedding` in their `modules.json` or 2) require calling `model.set_pooling_include_prompt(include_prompt=False)`.

### Scientific Publications
[SPECTER](https://arxiv.org/abs/2004.07180) is a model trained on scientific citations and can be used to estimate the similarity of two publications. We can use it to find similar papers.

- **allenai-specter** - [Semantic Search Python Example](https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic-search/semantic_search_publications.py) / [Semantic Search Colab Example](https://colab.research.google.com/drive/12hfBveGHRsxhPIUMmJYrll2lFU4fOX06)





### Natural Questions (NQ) Dataset Models
The following models were trained on [Google's Natural Questions dataset](https://ai.google.com/research/NaturalQuestions), a dataset with 100k real queries from Google search together with the relevant passages from Wikipedia.

- **nq-distilbert-base-v1**: MRR10: 72.36 on NQ dev set (small)

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("nq-distilbert-base-v1")

query_embedding = model.encode("How many people live in London?")

# The passages are encoded as [ [title1, text1], [title2, text2], ...]
passage_embedding = model.encode(
    [["London", "London has 9,787,426 inhabitants at the 2011 census."]]
)

print("Similarity:", util.cos_sim(query_embedding, passage_embedding))
```

You can index the passages as shown [here](../examples/applications/semantic-search/README.md).

**Note:** The NQ model doesn't perform well. Use the above mentioned Multi-QA models to achieve the optimal performance.

[More details](pretrained-models/nq-v1.md)



### DPR-Models

In [Dense Passage Retrieval  for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)  Karpukhin et al. trained models based on [Google's Natural Questions dataset](https://ai.google.com/research/NaturalQuestions):
- **facebook-dpr-ctx_encoder-single-nq-base** 
- **facebook-dpr-question_encoder-single-nq-base**

They also trained models on the combination of Natural Questions, TriviaQA, WebQuestions, and CuratedTREC.
- **facebook-dpr-ctx_encoder-multiset-base** 
- **facebook-dpr-question_encoder-multiset-base**

**Note:** The DPR models perform comparabily bad. Use the above mentioned Multi-QA models to achieve the optimal performance.

[More details & usage of the DPR models](pretrained-models/dpr.md)

### Average Word Embeddings Models

The following models apply compute the average word embedding for some well-known word embedding methods. Their computation speed is much higher than the transformer based models, but the quality of the embeddings are worse.
- **average_word_embeddings_glove.6B.300d**
- **average_word_embeddings_komninos**
- **average_word_embeddings_levy_dependency**
- **average_word_embeddings_glove.840B.300d**
