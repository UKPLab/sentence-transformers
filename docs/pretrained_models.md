# Pretrained Models

We provide various pre-trained models. Using these models is easy:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('model_name')
```

Alternatively, you can download and unzip them from [here](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/).

## Choosing the Right Model
Sadly there cannot exist a universal model that performs great on all possible tasks. Models strong on one task, will be weak for another task. Hence, it is important to select the right model for your task.


## Paraphrase Identification

The following models **are recommended for various applications**, as they were trained on Millions of paraphrase examples. They create extremely good results for various similarity and retrieval tasks. They are currently under development, better versions and more details will be released in future. But they many tasks they work better than the NLI / STSb models.

- **paraphrase-distilroberta-base-v1** - Trained on large scale paraphrase data.
- **paraphrase-xlm-r-multilingual-v1** - Multilingual version of distilroberta-base-paraphrase-v1, trained on parallel data for 50+ languages. 

## Semantic Textual Similarity
The following models were optimized for [Semantic Textual Similarity](usage/semantic_textual_similarity.md) (STS). They were trained on SNLI+MultiNLI and then fine-tuned on the STS benchmark train set.
 
 The best available models for STS are:
- **stsb-roberta-large** - STSb performance: 86.39
- **stsb-roberta-base** - STSb performance: 85.44
- **stsb-bert-large** - STSb performance: 85.29
- **stsb-distilbert-base** - STSb performance:  85.16

[» Full List of STS Models](https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0)



## Duplicate Questions Detection

The following models were trained for duplicate questions mining and duplicate questions retrieval. You can use them to detect duplicate questions in a large corpus (see [paraphrase mining](usage/paraphrase_mining.md)) or to search for similar questions (see [semantic search](usage/semantic_search.md)). 

Available models:
- **quora-distilbert-base** - Model first tuned on NLI+STSb data, then fine-tune for Quora Duplicate Questions detection retrieval.
- **quora-distilbert-multilingual** - Multilingual version of *distilbert-base-nli-stsb-quora-ranking*. Fine-tuned with parallel data for 50+ languages. 

## Information Retrieval 

The following models were trained on [MSMARCO Passage Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking): Given a search query (which can be anything like key words, a sentence, a question), find the relevant passages. You can index the embeddings and use it for dense information retrieval, outperforming lexical approaches like BM25.

- **msmarco-distilroberta-base-v2**: MRR@10: 28.55 on MS MARCO dev set
- **msmarco-roberta-base-v2**: MRR@10: 29.17 on MS MARCO dev set
- **msmarco-distilbert-base-v2**: MRR@10: 30.77 on MS MARCO  dev set

```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('msmarco-distilroberta-base-v2')

query_embedding = model.encode('How big is London')
passage_embedding = model.encode('London has 9,787,426 inhabitants at the 2011 census')

print("Similarity:", util.pytorch_cos_sim(query_embedding, passage_embedding))
```

You can index the passages as shown [here](https://www.sbert.net/docs/usage/semantic_search.html).

[More details](pretrained-models/msmarco-v2.md)


## Multi-Lingual Models
The following models generate aligned vector spaces, i.e., similar inputs in different languages are mapped close in vector space. You do not need to specify the input language.  Details are in our publication [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813):

Currently, there are models for two use-cases: 

**Semantic Similarity**

These models find semantically similar sentences within one language or across languages:

- **distiluse-base-multilingual-cased-v2**: Multilingual knowledge distilled version of [multilingual Universal Sentence Encoder](https://arxiv.org/abs/1907.04307). While the original mUSE model only supports 16 languages, this multilingual knowledge distilled version supports 50+ languages.
- **paraphrase-xlm-r-multilingual-v1** - Multilingual version of distilroberta-base-paraphrase-v1, trained on parallel data for 50+ languages. 
- **stsb-xlm-r-multilingual**: Produces similar embeddings as the bert-base-nli-stsb-mean-token model. Trained on parallel data for 50+ languages.
- **quora-distilbert-multilingual** - Multilingual version of *distilbert-base-nli-stsb-quora-ranking*.  Fine-tuned with parallel data for 50+ languages. 
- **T-Systems-onsite/cross-en-de-roberta-sentence-transformer** - Multilingual model for English an German. [[More]](https://huggingface.co/T-Systems-onsite/cross-en-de-roberta-sentence-transformer)

**Bitext Mining** 

Bitext mining describes the process of finding translated sentence pairs in two languages. If this is your use-case, the following model gives the best performance:
- **LaBSE** - [LaBSE](https://arxiv.org/abs/2007.01852) Model. Supports 109 languages. Works well for finding translation pairs in multiple languages. As detailed  [here](https://arxiv.org/abs/2004.09813), LaBSE works less well for assessing the similarity of sentence pairs that are not translations of each other.



---

XLM-R models support the following 100 languages.

 Language | Language|Language |Language | Language
---|---|---|---|---
Afrikaans | Albanian | Amharic | Arabic | Armenian 
Assamese | Azerbaijani | Basque | Belarusian | Bengali 
Bengali Romanize | Bosnian | Breton | Bulgarian | Burmese 
Burmese zawgyi font | Catalan | Chinese (Simplified) | Chinese (Traditional) | Croatian 
Czech | Danish | Dutch | English | Esperanto 
Estonian | Filipino | Finnish | French | Galician
Georgian | German | Greek | Gujarati | Hausa
Hebrew | Hindi | Hindi Romanize | Hungarian | Icelandic
Indonesian | Irish | Italian | Japanese | Javanese
Kannada | Kazakh | Khmer | Korean | Kurdish (Kurmanji)
Kyrgyz | Lao | Latin | Latvian | Lithuanian
Macedonian | Malagasy | Malay | Malayalam | Marathi
Mongolian | Nepali | Norwegian | Oriya | Oromo
Pashto | Persian | Polish | Portuguese | Punjabi
Romanian | Russian | Sanskrit | Scottish Gaelic | Serbian
Sindhi | Sinhala | Slovak | Slovenian | Somali
Spanish | Sundanese | Swahili | Swedish | Tamil
Tamil Romanize | Telugu | Telugu Romanize | Thai | Turkish
Ukrainian | Urdu | Urdu Romanize | Uyghur | Uzbek
Vietnamese | Welsh | Western Frisian | Xhosa | Yiddish

We used the following languages for [Multilingual Knowledge Distillation](https://arxiv.org/abs/2004.09813): ar, bg, ca, cs, da, de, el, es, et, fa, fi, fr, fr-ca, gl, gu, he, hi, hr, hu, hy, id, it, ja, ka, ko, ku, lt, lv, mk, mn, mr, ms, my, nb, nl, pl, pt, pt, pt-br, ro, ru, sk, sl, sq, sr, sv, th, tr, uk, ur, vi, zh-cn, zh-tw. 

Extending a model to new languages is easy by following [the description here](https://www.sbert.net/examples/training/multilingual/README.html).


## Scientific Publications
[SPECTER](https://github.com/allenai/specter) is a model trained on scientific citations and can be used to estimate the similarity of two publications. We can also use it to find similar papers.

- **allenai-specter** - [Semantic Search Python Example](https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic-search/semantic_search_publications.py) / [Semantic Search Colab Example](https://colab.research.google.com/drive/12hfBveGHRsxhPIUMmJYrll2lFU4fOX06?usp=sharing)


## Average Word Embeddings Models

The following models apply compute the average word embedding for some well-known word embedding methods. Their computation speed is much higher than the transformer based models, but the quality of the embeddings are worse.
- **average_word_embeddings_glove.6B.300d**
- **average_word_embeddings_komninos**
- **average_word_embeddings_levy_dependency**
- **average_word_embeddings_glove.840B.300d**
