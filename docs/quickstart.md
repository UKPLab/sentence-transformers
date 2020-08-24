# Quickstart
Once you have SentenceTransformers [installed](installation.md), the usage is simple:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

#Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']

#Sentences are encoded by calling model.encode()
sentence_embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
```


With `SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')` we define which sentence transformer model we like to load. In this example, we load *distilbert-base-nli-stsb-mean-tokens*, which is a DistilBERT-base-uncased model fine tuned on Natural Language Inference (NLI) and Semantic Textual Similarity Benchmark (STSb) data. 

BERT (and other transformer networks) output for each token in our input text an embedding. In order to create a fixed-sized sentence embedding out of this, the model applies mean pooling, i.e., the output embeddings for all tokens are averaged to yield a 768-dimensional vector.


## Pre-Trained Models (English)
Various pre-trained models exists optimized for many tasks exists. For a full list, see **[Full List of Pretrained Models](pretrained_models.md)**. To highlight some models:

**Semantic Similarity**:

The following models were optimized to assign semantic similarity scores with cosine-similarity for sentence pairs: 
- **distilbert-base-nli-stsb-mean-tokens**: DistilBERT-base-model (STSbenchmark Spearman correlation: 85.16)
- **roberta-base-nli-stsb-mean-tokens**: RoBERTa-base model (STSbenchmark Spearman correlation:: 85.44)
- **roberta-large-nli-stsb-mean-tokens**: RoBERTa-large model (STSbenchmark Spearman correlation:: 86.39)


**Duplicate Questions:**

The following models were trained on the [Quora Duplicate Questions](training/use_case/quora_duplicate_questions.md) dataset and are especially suitable for duplicate questions mining (given large set of questions, find all duplicates) and related questions semantic search (given new question, search large corpus for similar questions).
- **distilbert-base-nli-stsb-quora-ranking**: DistilBERT-base-model

**Average Word Embeddings:**

The following models perform a simple average word embeddings encoding.
- **average_word_embeddings_glove.6B.300d**: GloVe embeddings trained on Wikipedia
- **average_word_embeddings_glove.840B.300d**: GloVe embeddings traind on Common Crawl
- **average_word_embeddings_komninos**: Embeddings from Komninos et al.
- **average_word_embeddings_levy_dependency**: Dependency based embeddings from Levy et al.


## Multi-Lingual Pre-Trained Models 
In our publication [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813) I describe a method to extend mono-lingual sentence embeddings for many languages. An increasing number of models are currently extended to various languages.

Currently available models:
- **distiluse-base-multilingual-cased**: Supported languages: Arabic, Chinese, Dutch, English, French, German, Italian, Korean, Polish, Portuguese, Russian, Spanish, Turkish. Performance on the extended STS2017 dataset: 80.1
- **xlm-r-base-en-ko-nli-ststb**: Supported languages: English, Korean. Performance on Korean STSbenchmark: 81.47
- **xlm-r-large-en-ko-nli-ststb**: Supported languages: English, Korean. Performance on Korean STSbenchmark: 84.05

## Applications & Use-Cases

## Training your own Embeddings

Training your own sentence embeddings models for all type of use-cases is easy and requires often only minimal coding effort. For a comprehensive tutorial, see [Training/Overview](training/overview.md).

You can also extend easily existent sentence embeddings models to various languages from all types of language families.  For details, see [Multi-Lingual Training](training/multi_lingual_training.md).