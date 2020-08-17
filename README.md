# Sentence Transformers: Multilingual Sentence Embeddings using BERT / RoBERTa / XLM-RoBERTa & Co. with PyTorch
BERT / RoBERTa / XLM-RoBERTa produces out-of-the-box rather bad sentence embeddings. This repository fine-tunes BERT / RoBERTa / DistilBERT / ALBERT / XLNet with a siamese or triplet network structure to produce semantically meaningful sentence embeddings that can be used in unsupervised scenarios: Semantic textual similarity via cosine-similarity, clustering, semantic search.


We provide an increasing number of **state-of-the-art pretrained models** that can be used to derive sentence embeddings. See [Pretrained Models](#pretrained-models). Details of the implemented approaches can be found in our publication: [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) (EMNLP 2019).


You can use this code to easily **train your own sentence embeddings**, that are tuned for your specific task. We provide various dataset readers and you can tune sentence embeddings with different loss function, depending on the structure of your dataset. For further details, see [Train your own Sentence Embeddings](#Training).



## Setup
We recommend Python 3.6 or higher. The model is implemented with PyTorch (at least 1.2.0) using [transformers v3.0.2](https://github.com/huggingface/transformers).
The code does **not** work with Python 2.7.

**With pip**

Install the model with `pip`:
```
pip install -U sentence-transformers
```

**From source**

Clone this repository and install it with `pip`:
````
pip install -e .
```` 



## Getting Started

### Sentences Embedding with a Pretrained Model
[This example](https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/basic_embedding.py) shows you how to use an already trained Sentence Transformer model to embed sentences for another task.

First download a pretrained model.
````python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
````
Then provide some sentences to the model.
````python
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']
sentence_embeddings = model.encode(sentences)
````
And that's it already. We now have a list of numpy arrays with the embeddings.
````python
for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
````

## Training
This framework allows you to fine-tune your own sentence embedding methods, so that you get task-specific sentence embeddings. You have various options to choose from in order to get perfect sentence embeddings for your specific task. 

### Dataset Download
First, you should download some datasets. For this run the [examples/datasets/get_data.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/datasets/get_data.py):
```
python examples/datasets/get_data.py
```

It will download some [datasets](https://github.com/UKPLab/sentence-transformers/blob/master/examples/datasets) and store them on your disk.


### Model Training from Scratch
[training_nli.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/training_transformers/training_nli.py) fine-tunes BERT (and other transformer models) from the pre-trained model as provided by Google & Co. It tunes the model on Natural Language Inference (NLI) data. Given two sentences, the model should classify if these two sentence entail, contradict, or are neutral to each other. For this, the two sentences are passed to a transformer model to generate fixed-sized sentence embeddings. These sentence embeddings are then passed to a softmax classifier to derive the final label (entail, contradict, neutral). This generates sentence embeddings that are useful also for other tasks like clustering or semantic textual similarity.


First, we define a sequential model of how a sentence is mapped to a fixed size sentence embedding:
```python
# Use BERT for mapping tokens to embeddings
word_embedding_model = models.Transformer('bert-base-uncased')

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
```

First, we use the BERT model (instantiated from bert-base-uncased) to map tokens in a sentence to the output embeddings from BERT. The next layer in our model is a Pooling model: In that case, we perform mean-pooling. You can also perform max-pooling or use the embedding from the CLS token. You can also combine multiple poolings together.

These two modules (word_embedding_model and pooling_model) form our SentenceTransformer. Each sentence is now passed first through the word_embedding_model and then through the pooling_model to give fixed sized sentence vectors.


Next, we specify a train dataloader:
```python
nli_reader = NLIDataReader('datasets/AllNLI')

train_data = SentencesDataset(nli_reader.get_examples('train.gz'), model=model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)
```

The `NLIDataReader` reads the AllNLI dataset and we generate a dataloader that is suitable for training the Sentence Transformer model. As training loss, we use a Softmax Classifier.

Next, we also specify a dev-set. The dev-set is used to evaluate the sentence embedding model on some unseen data. Note, the dev-set can be any data, in this case, we evaluate on the dev-set of the STS benchmark dataset.  The `evaluator` computes the performance metric, in this case, the cosine-similarity between sentence embeddings are computed and the Spearman-correlation to the gold scores is computed.

```python
sts_reader = STSBenchmarkDataReader('datasets/stsbenchmark')
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(sts_reader.get_examples('sts-dev.csv'))
```

 The training then looks like this:
 ```python
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )
```



### Continue Training on Other Data
[training_stsbenchmark_continue_training.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/training_transformers/training_stsbenchmark_continue_training.py) shows an example where training on a fine-tuned model is continued. In that example, we use a sentence transformer model that was first fine-tuned on the NLI dataset and then continue training on the training data from the STS benchmark.

First, we load a pre-trained model from the server:
```python
model = SentenceTransformer('bert-base-nli-mean-tokens')
```


The next steps are as before. We specify training and dev data:
```python
sts_reader = STSBenchmarkDataReader('datasets/stsbenchmark', normalize_scores=True)
train_data = SentencesDataset(sts_reader.get_examples('sts-train.csv'), model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(sts_reader.get_examples('sts-dev.csv'))
```

In that example, we use CosineSimilarityLoss, which computes the cosine similarity between two sentences and compares this score with a provided gold similarity score.

Then we can train as before:
```python
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)
```


## Loading SentenceTransformer Models
Loading trained models is easy. You can specify a path:
```python
model = SentenceTransformer('./my/path/to/model/')
```
Note: It is important that a / or \ is the path, otherwise, it is not recognized as a path.

You can also host the training output on a server and download it:
 ```python
model = SentenceTransformer('http://www.server.com/path/to/model/my_model.zip')
```
With the first call, the model is downloaded and stored in the local torch cache-folder (`~/.cache/torch/sentence_transformers`). In order to work, you must zip all files and subfolders of your model. 

We also provide several pre-trained models, that can be loaded by just passing a name:

 ```python
model = SentenceTransformer('bert-base-nli-mean-tokens')
```

This downloads the `bert-base-nli-mean-tokens` from our server and stores it locally.

## Loading custom BERT models
If you have fine-tuned BERT (or similar models) and you want to use it to generate sentence embeddings, you must construct an appropriate sentence transformer model from it. This is possible by using this code:

```python  
from sentence_transformers import models
# Use BERT for mapping tokens to embeddings
word_embedding_model = models.Transformer('path/to/your/BERT/model')

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
```

## Training Multilingual Sentence Embeddings Models
We provide code and example to easily train sentence embedding models for various languages and also port existent sentence embedding models to new languages. For details, see [multilingual-models.md](https://github.com/UKPLab/sentence-transformers/blob/master/docs/training/multilingual-models.md) and our publication [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813).

## Pretrained Models
We provide the following models. You can use them in the following way:
 ```python
model = SentenceTransformer('name_of_model')
```



### English Pre-Trained Models
In the following you find selected models that were trained on English data only. For the full list of available models, see [SentenceTransformer Pretrained Models](https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit?usp=sharing). See the next section for multi-lingual models.

**Trained on NLI data**

These models were trained on SNLI and MultiNLI dataset to create universal sentence embeddings. For more details, see: [nli-models.md](https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained-models/nli-models.md).
- **bert-base-nli-mean-tokens**: BERT-base model with mean-tokens pooling. Performance: STSbenchmark: 77.12
- **bert-large-nli-mean-tokens**: BERT-large with mean-tokens pooling. Performance: STSbenchmark: 79.19
- **roberta-base-nli-mean-tokens**: RoBERTa-base with mean-tokens pooling. Performance: STSbenchmark: 77.49
- **roberta-large-nli-mean-tokens**: RoBERTa-base with mean-tokens pooling. Performance: STSbenchmark: 78.69
- **distilbert-base-nli-mean-tokens**: DistilBERT-base with mean-tokens pooling. Performance: STSbenchmark: 76.97


**Trained on STS data**

These models were first fine-tuned on the AllNLI datasent, then on train set of STS benchmark. They are specifically well suited for semantic textual similarity. For more details, see: [sts-models.md](https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained-models/sts-models.md).
- **bert-base-nli-stsb-mean-tokens**: Performance: STSbenchmark: 85.14
- **bert-large-nli-stsb-mean-tokens**: Performance: STSbenchmark: 85.29
- **roberta-base-nli-stsb-mean-tokens**: Performance: STSbenchmark: 85.44
- **roberta-large-nli-stsb-mean-tokens**: Performance: STSbenchmark: 86.39
- **distilbert-base-nli-stsb-mean-tokens**: Performance: STSbenchmark: 84.38


### Multilingual Models
The following models can be used for languages other than English. The vector spaces for the included languages are aligned, i.e., two sentences are mapped to the same point in vector space independent of the language. The models can be used for cross-lingual tasks. For more details see [multilingual-models.md](https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained-models/multilingual-models.md).

- **distiluse-base-multilingual-cased**: Supported languages: Arabic, Chinese, Dutch, English, French, German,  Italian, Korean, Polish, Portuguese, Russian, Spanish, Turkish. Performance on the extended STS2017: 80.1
- **xlm-r-100langs-bert-base-nli-mean-tokens**: Produces similar embeddings as the bert-base-nli-mean-token model for 100+ languages
- **xlm-r-100langs-bert-base-nli-stsb-mean-tokens**: Produces similar embeddings as the bert-base-nli-stsb-mean-token model for 100+ languages


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


## Performance

Extensive evaluation is currently undergoing, but here we provide some preliminary results.

| Model    | STS benchmark | SentEval  |
| ----------------------------------|:-----: |:---:   |
| Avg. GloVe embeddings             | 58.02  | 81.52  |
| BERT-as-a-service avg. embeddings | 46.35  | 84.04  |
| BERT-as-a-service CLS-vector      | 16.50  | 84.66  |
| InferSent - GloVe                 | 68.03  | 85.59  |
| Universal Sentence Encoder        | 74.92  | 85.10  |
|**Sentence Transformer Models**    ||
| bert-base-nli-mean-tokens         | 77.12  | 86.37 |
| bert-large-nli-mean-tokens        | 79.19  | 87.78 |
| bert-base-nli-stsb-mean-tokens    | 85.14  | 86.07 |
| bert-large-nli-stsb-mean-tokens   | 85.29 | 86.66|


## Loss Functions
We implemented various loss-functions that allow training of sentence embeddings from various datasets. These loss-functions are in the package `sentence_transformers.losses`.
 
- *SoftmaxLoss*: Given the sentence embeddings of two sentences, trains a softmax-classifier. Useful for training on datasets like NLI.
- *CosineSimilarityLoss*: Given a sentence pair and a gold similarity score (either between -1 and 1 or between 0 and 1), computes the cosine similarity between the sentence embeddings and minimizes the mean squared error loss.
- *TripletLoss*: Given a triplet (anchor, positive example, negative example), minimizes the [triplet loss](https://en.wikipedia.org/wiki/Triplet_loss).
- *BatchHardTripletLoss*: Implements the *batch hard triplet loss* from the paper [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737). Each batch must contain multiple examples from the same class. The loss optimizes then the distance between the most-distance positive pair and the closest negative-pair.
- *MultipleNegativesRankingLoss*: Each batch has one positive pair, all other pairs are treated as negative examples. The loss was used in the papers [Efficient Natural Language Response Suggestion for Smart Reply](https://arxiv.org/pdf/1705.00652.pdf) and [Learning Cross-Lingual Sentence Representations via a Multi-task Dual-Encoder Model](https://arxiv.org/pdf/1810.12836.pdf).

## Models
This framework implements various modules, that can be used sequentially to map a sentence to a sentence embedding. The different modules can be found in the package `sentence_transformers.models`. Each pipeline consists of the following modules.


**Word Embeddings:** These models map tokens to token embeddings.
- **[Transformer](sentence_transformers/models/Transformer.py)**: You can use any huggingface [pretrained models](https://huggingface.co/transformers/pretrained_models.html) including BERT, RoBERTa, DistilBERT, ALBERT, XLNet, XLM-RoBERTa, ELECTRA, FlauBERT, CamemBERT... 
- **[WordEmbeddings](sentence_transformers/models/WordEmbeddings.py)**: Uses traditional word embeddings like word2vec or GloVe to map tokens to vectors. Example: [training_stsbenchmark_avg_word_embeddings.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/training_basic_models/training_stsbenchmark_avg_word_embeddings.py)

**Embedding Transformations:** These models transform token embeddings in some way
- **[LSTM](sentence_transformers/models/LSTM.py)**: Runs a bidirectional LSTM. Example: [training_stsbenchmark_bilstm.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/training_basic_models/training_stsbenchmark_bilstm.py).
- **[CNN](sentence_transformers/models/CNN.py)**: Runs a CNN model with multiple kernel sizes. Example: [training_stsbenchmark_cnn.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/training_basic_models/training_stsbenchmark_cnn.py).
- **[WordWeights](sentence_transformers/models/WordWeights.py)**: This model can be used after WordEmbeddings and before Pooling to apply a weighting to the token embeddings, for example, a tf-idf weighting. Example: [training_stsbenchmark_tf-idf_word_embeddings.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/training_basic_models/training_stsbenchmark_tf-idf_word_embeddings.py).
- **[Pooling](sentence_transformers/models/Pooling.py)**: After tokens are mapped to embeddings, we apply the pooling, where you can compute a mean/max-pooling or use the CLS-token embedding (for BERT and XLNet). You can also combine multiple poolings.
- **[WeightedLayerPooling](sentence_transformers/models/WeightedLayerPooling.py)**: Learns a weighted pooling of all hidden layer of transformer models like BERT. Requires that the model has set output_hidden_states to true.
- **[WKPooling](sentence_transformers/models/WKPooling.py)**: Pooling based on the paper of *[SBERT-WK](https://arxiv.org/abs/2002.06652)*. Note, WKPooling uses QR decomposition which must run on the CPU. This makes the pooling rather slow. For some models, WKPooling leads to a performance improvement.

**Sentence Embeddings Models:** These models map a sentence directly to a fixed size sentence embedding:
- **[BoW](sentence_transformers/models/BoW.py)**: Computes a fixed size bag-of-words (BoW) representation of the input text. Can be initialized with IDF-values to create a tf-idf vector. Note that this model is not trainable. Example: [training_stsbenchmark_bow.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/training_basic_models/training_stsbenchmark_bow.py)


**Sentence Embeddings Transformations:** These models can be added once we have a fixed size sentence embedding.
- **[Dense](sentence_transformers/models/Pooling.py)**: A fully-connected feed-forward network to create a Deep Averaging Network (DAN). You can stack multiple Dense models. Example: [training_stsbenchmark_avg_word_embeddings.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/training_basic_models/training_stsbenchmark_avg_word_embeddings.py)



## Multitask Training
This code allows multi-task learning with training data from different datasets and with different loss-functions. For an example, see [training_multi-task.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/training_transformers/training_multi-task.py).


## Application Examples
We present some examples, how the generated sentence embeddings can be used for downstream applications.

### Semantic Search
Semantic search is the task of finding similar sentences to a given sentence. See [semantic_search.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic_search.py).

We first generate an embedding for all sentences in a corpus:
```python
embedder = SentenceTransformer('bert-base-nli-mean-tokens')

# Corpus with example sentences
corpus = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'A cheetah is running behind its prey.']

corpus_embeddings = embedder.encode(corpus)
```

Then, we generate the embeddings for different query sentences:
```python
queries = ['A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.', 'A cheetah chases prey on across a field.']
query_embeddings = embedder.encode(queries)
```

We then use scipy to find the most-similar embeddings for queries in the corpus:
```python
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
```

The output looks like this:
```
Query: A man is eating pasta.
Top 5 most similar sentences in corpus:
A man is eating a piece of bread. (Score: 0.8518)
A man is eating a food. (Score: 0.8020)
A monkey is playing drums. (Score: 0.4167)
A man is riding a horse. (Score: 0.2621)
A man is riding a white horse on an enclosed ground. (Score: 0.2379)


Query: Someone in a gorilla costume is playing a set of drums.
Top 5 most similar sentences in corpus:
A monkey is playing drums. (Score: 0.8514)
A man is eating a piece of bread. (Score: 0.3671)
A man is eating a food. (Score: 0.3559)
A man is riding a horse. (Score: 0.3153)
The girl is carrying a baby. (Score: 0.2589)


Query: A cheetah chases prey on across a field.
Top 5 most similar sentences in corpus:
A cheetah is running behind its prey. (Score: 0.9073)
Two men pushed carts through the woods. (Score: 0.3896)
A man is riding a horse. (Score: 0.3789)
A man is riding a white horse on an enclosed ground. (Score: 0.3544)
A monkey is playing drums. (Score: 0.3435)

```


### Clustering
[clustering.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/clustering.py) depicts an example to cluster similar sentences based on their sentence embedding similarity.

As before, we first compute an embedding for each sentence:
```python
embedder = SentenceTransformer('bert-base-nli-mean-tokens')

# Corpus with example sentences
corpus = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'A man is eating pasta.',
          'The girl is carrying a baby.',
          'The baby is carried by the woman',
          'A man is riding a horse.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'Someone in a gorilla costume is playing a set of drums.',
          'A cheetah is running behind its prey.',
          'A cheetah chases prey on across a field.']

corpus_embeddings = embedder.encode(corpus)
```

Then, we perform k-means clustering using sklearn:
```python
from sklearn.cluster import KMeans

num_clusters = 5
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_
```

The output looks like this:
```
Cluster  1
['The girl is carrying a baby.', 'The baby is carried by the woman']

Cluster  2
['A cheetah is running behind its prey.', 'A cheetah chases prey on across a field.']

Cluster  3
['A monkey is playing drums.', 'Someone in a gorilla costume is playing a set of drums.']

Cluster  4
['A man is eating food.', 'A man is eating a piece of bread.', 'A man is eating pasta.']

Cluster  5
['A man is riding a horse.', 'A man is riding a white horse on an enclosed ground.']
```

## Citing & Authors
If you find this repository helpful, feel free to cite our publication [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084):
``` 
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/1908.10084",
}
```


If you use the code for multilingual models, feel free to cite our publication [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813):
``` 
@article{reimers-2020-multilingual-sentence-bert,
    title = "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation",
    author = "Reimers, Nils and Gurevych, Iryna",
    journal= "arXiv preprint arXiv:2004.09813",
    month = "04",
    year = "2020",
    url = "http://arxiv.org/abs/2004.09813",
}
```


The main contributors of this repository are:
- [Nils Reimers](https://github.com/nreimers)
- [Gregor Geigle](https://github.com/aaronsom)

Contact person: Nils Reimers, info@nils-reimers.de

https://www.ukp.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.







