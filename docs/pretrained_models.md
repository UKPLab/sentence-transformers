# Pretrained Models

We provide various pre-trained models. Using these models is easy:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('model_name')
```

Alternatively, you can download and unzip them from [here](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/).

## Choosing the Right Model
Sadly there cannot exist a universal model that performs great on all possible tasks. Models strong on one task, will be weak for another task. Hence, it is important to select the right model for your task.


## Semantic Textual Similarity
The following models were optimized for [Semantic Textual Similarity](usage/semantic_textual_similarity) (STS): You can compute the embeddings for two sentences and then use cosine-similarity to get a score -1 ... 1 to indicate their semantic similarity. They were trained on [NLI](pretrained-models/nli-models.md) and [STS](pretrained-models/sts-models.md) data. They are evaluated on the STSbenchmark dataset.

We can recommend this models as general purpose models. The best available models are:
- **roberta-large-nli-stsb-mean-tokens** - STSb performance: 86.39
- **roberta-base-nli-stsb-mean-tokens** - STSb performance: 85.44
- **bert-large-nli-stsb-mean-tokens** - STSb performance: 85.29
- **distilbert-base-nli-stsb-mean-tokens** - STSb performance:  85.16

[» Full List of STS Models](https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0)

I can recommend the **distilbert-base-nli-stsb-mean-tokens** model, which gives a nice balance between speed and performance.

## Multi-Lingual Models
The following models generate aligned vector spaces, i.e., similar inputs in different languages are mapped close in vector space. You do not need to specify the input language.  Details are in our publication [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813):

- **distiluse-base-multilingual-cased**: Supported languages: Arabic, Chinese, Dutch, English, French, German,  Italian, Korean, Polish, Portuguese, Russian, Spanish, Turkish. Performance on the extended STS2017: 80.1
- **xlm-r-100langs-bert-base-nli-mean-tokens**: Produces similar embeddings as the bert-base-nli-mean-token model for 100+ languages
- **xlm-r-100langs-bert-base-nli-stsb-mean-tokens**: Produces similar embeddings as the bert-base-nli-stsb-mean-token model for 100+ languages
- **distilbert-multilingual-nli-stsb-quora-ranking** - Multilingual version of *distilbert-base-nli-stsb-quora-ranking*. DistilBERT-multilingual support 104 languages. Fine-tuned with parallel data for 50+ languages. 


XLM-R supports the following 100 languages.

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

The XLM-R-100langs models were fine-tuned using [Multilingual Knowledge Distillation](https://arxiv.org/abs/2004.09813) using parallel data for the following languages: ar, bg, ca, cs, da, de, el, es, et, fa, fi, fr, fr-ca, gl, gu, he, hi, hr, hu, hy, id, it, ja, ka, ko, ku, lt, lv, mk, mn, mr, ms, my, nb, nl, pl, pt, pt, pt-br, ro, ru, sk, sl, sq, sr, sv, th, tr, uk, ur, vi, zh-cn, zh-tw. It achieves also quite good performance scores for languages not in these lists.


## Duplicate Questions Detection

The following models were trained duplicate questions mining and duplicate questions retrieval. You can use them to detect duplicate questions in a large corpus (see [paraphrase mining](usage/paraphrase_mining) ) or to search for similar questions (see [semantic search](usage/semantic_search)). 

Available models:
- **distilbert-base-nli-stsb-quora-ranking** - Model first tuned on NLI+STSb data, then fine-tune for Quora Duplicate Questions detection retrieval.
- **distilbert-multilingual-nli-stsb-quora-ranking** - Multilingual version of *distilbert-base-nli-stsb-quora-ranking*. DistilBERT-multilingual support 104 languages. Fine-tuned with parallel data for 50+ languages. 


## Wikipedia Sections
The following models is trained on the dataset from Dor et al. 2018, [Learning Thematic Similarity Metric Using Triplet Networks](https://aclweb.org/anthology/P18-2009) and learns if two sentences belong to the same section in a Wikipedia page or not. It can be used to do fine-grained clustering of similar sentences into sections / topics. [Further details](pretrained-models/wikipedia-sections-models.md)    

- **bert-base-wikipedia-sections-mean-tokens**: 80.42% accuracy on Wikipedia Triplets test set.

## Average Word Embeddings Models

The following models apply compute the average word embedding for some well-known word embedding methods. Their computation speed is much higher than the transformer based models, but the quality of the embeddings are worse.
- **average_word_embeddings_glove.6B.300d**
- **average_word_embeddings_komninos**
- **average_word_embeddings_levy_dependency**
- **average_word_embeddings_glove.840B.300d**
