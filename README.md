# Sentence Transformers: Sentence Embeddings using BERT / XLNet with PyTorch
BERT / XLNet produces out-of-the-box rather bad sentence embeddings. This repository fine-tunes BERT / XLNet with a siamese or triplet network structure to produce semantically meaningful sentence embeddings that can be used in unsupervised scenarios: Semantic textual similarity via cosine-similarity, clustering, semantic search.


We provide an increasing number of **state-of-the-art pretrained models** that can be used to derive sentence embeddings. See [Pretrained Models](#pretrained-models).


You can use this code to easily **train your own sentence embeddings**, that are tuned for your specific task. We provide various dataset readers and you can tune sentence embeddings with different loss function, depending on the structure of your dataset. For further details, see [Train your own Sentence Embeddings](#Training).



## Setup
We recommend Python 3.6 or higher. The model is implemented with PyTorch (at least 1.0.1) using [pytorch-transformers v1.1.0](https://github.com/huggingface/pytorch-transformers).
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
[This example](examples/basic_embedding.py) shows you how to use an already trained Sentence Transformer model to embed sentences for another task.

First download a pretrained model.
````
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
````
Then provide some sentences to the model.
````
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']
sentence_embeddings = model.encode(sentences)
````
And that's it already. We now have a list of numpy arrays with the embeddings.
````
for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
````

## Training
This framework allows you to fine-tune your own sentence embedding methods, so that you get task-specific sentence embeddings. You have various options to choose from in order to get perfect sentence embeddings for your specific task. 

### Dataset Download
First, you should download some datasets. For this run the [examples/datasets/get_data.py](examples/datasets/get_data.py):
```
python examples/datasets/get_data.py
```

It will download some [datasets](examples/datasets) and store it on your disk.


### Model Training from Scratch
[examples/training_nli.py](examples/training_nli.py) fine-tunes BERT from the pre-trained model as provided by Google. It tunes the model on Natural Language Inference (NLI) data. Given two sentences, the model should classify if these two sentence entail, contradict, or are neutral to each other. For this, the two sentences are passed to a transformer model to generate fixed-sized sentence embeddings. These sentence embeddings are then passed to a softmax classifier to derive the final label (entail, contradict, neutral). This generates sentence embeddings that are useful also for other tasks like clustering or semantic textual similarity.


First, we define a sequential model how a sentence is mapped to a fixed size sentence embedding:
```
# Use BERT for mapping tokens to embeddings
word_embedding_model = models.BERT('bert-base-uncased')

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
```

First, we use the BERT model (instantiated from bert-base-uncased) to map tokens in a sentence to the output embeddings from BERT. The next layer in our model is a Pooling model: In that case, we perform mean-pooling. You can also perform max-pooling or use the embedding from the CLS token. You can also combine multiple poolings together.

These two modules (word_embedding_model and pooling_model) form our SentenceTransformer. Each sentence is now passed first through the word_embedding_model and then through the pooling_model to give fixed sized sentence vectors.


Next, we specify a train dataloader:
```
nli_reader = NLIDataReader('datasets/AllNLI')

train_data = SentencesDataset(nli_reader.get_examples('train.gz'), model=model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)
```

The `NLIDataReader` reads the AllNLI dataset and we generate a dataload that is suitable for training the Sentence Transformer model. As training loss, we use a Softmax Classifier.

Next, we also specify a dev-set. The dev-set is used, to evaluate the sentence embeddings model on some unseen data. Note, the dev-set can be any data, in this case, we evaluate on the dev-set of the STS benchmark dataset.  The `evaluator` computes the performance metric, in this case, the cosine-similarity between sentence embeddings are computed and the Spearman-correlation to the gold scores is computed.

```
sts_reader = STSDataReader('datasets/stsbenchmark')
dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)
```

 The training then looks like this:
 ```
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )
```



### Continue Training on Other Data
[examples/training_stsbenchmark.py](examples/training_stsbenchmark.py) shows an example, where training on a fine-tuned model is continued. In that example, we use a sentence transformer model that was first fine-tuned on the NLI dataset, and then continue training on the training data from the STS benchmark.

First, we load a pre-trained model from the server:
```
model = SentenceTransformer('bert-base-nli-mean-tokens')
```


The next steps are as before. We specify training and dev data:
```
sts_reader = STSDataReader('datasets/stsbenchmark', normalize_scores=True)
train_data = SentencesDataset(sts_reader.get_examples('sts-train.csv'), model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)
```

In that example, we use CosineSimilarityLoss, which computes the cosine similarity between two sentences and compares this score with a provided gold similarity score.

Then we can train as before:
```
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)
```


## Load Models
Loading trained models is easy. You can specify a path:
```
model = SentenceTransformer('./my/path/to/model/')
```
Note: It is important that a / or \ is the path, otherwise, it is not recognize as a path.

You can also host the training output on a server and download it:
 ```
model = SentenceTransformer('http://www.server.com/path/to/model/my_model.zip')
```
With the first call, the model is downloaded and stored in the local torch cache-folder (`~/.cache/torch/sentence_transformers`). In order to work, you must zip all files and subfolders of your model. 

We also provide several pre-trained models, that can be loaded by just passing a name:

 ```
model = SentenceTransformer('bert-base-nli-mean-tokens')
```

This downloads the `bert-base-nli-mean-tokens` from our server and stores it locally.

## Pretrained Models
We provide the following models. You can use them in the following way:
 ```
model = SentenceTransformer('name_of_model')
```

The list is increasing as soon was new models increasing.

### Sentence Embeddings using BERT
BERT Sentence Embeddings have been extensively tested and tuned. We released the following pre-trained model for your usage:

**Trained on NLI data**

These models were trained on SNLI and MultiNLI dataset to create universal sentence embeddings. For more details, see: [nli-models.md](docs/pretrained-models/nli-models.md).
- **bert-base-nli-mean-tokens**: This model fine-tuned BERT-base on the AllNLI dataset. As pooling strategy, mean-tokens was used. Performance: STSbenchmark: 77.12
- **bert-base-nli-max-tokens**: This model fine-tuned BERT-base on the AllNLI dataset. As pooling strategy, max-tokens was used. Performance: STSbenchmark: 77.18
- **bert-base-nli-cls-token**: This model fine-tuned BERT-base on the AllNLI dataset. As pooling strategy, the CLS token was used. Performance: STSbenchmark: 76.30
- **bert-large-nli-mean-tokens**: This model fine-tuned BERT-large on the AllNLI dataset. As pooling strategy, mean-tokens was used. Performance: STSbenchmark: 79.19
- **bert-large-nli-max-tokens**: This model fine-tuned BERT-large on the AllNLI dataset. As pooling strategy, max-tokens was used. Performance: STSbenchmark: 78.32
- **bert-large-nli-cls-token**: This model fine-tuned BERT-large on the AllNLI dataset. As pooling strategy, the CLS token was used. Performance: STSbenchmark: 78.29

**Trained on STS data**

These models were fine-tuned on the training set of the STS benchmark. They are specifically well suited for semantic textual similarity. For more details, see: [sts-models.md](docs/pretrained-models/sts-models.md).
- **bert-base-nli-stsb-mean-tokens**: First fine-tuned on AllNLI, then on STS benchmark training set. Performance: STSbenchmark: 85.14
- **bert-large-nli-stsb-mean-tokens**: First fine-tuned on AllNLI, then on STS benchmark training set. Performance: STSbenchmark: 85.29

**Trained on Wikipedia Sections Triplets**

These models were fine-tuned on triplets generated from Wikipedia sections. These models work well if fine-grained clustering of sentences on a similar topic are required. For more details, see: [wikipedia-sections-models.md](docs/pretrained-models/wikipedia-sections-models.md).
- **bert-base-wikipedia-sections-mean-tokens**: 80.42% accuracy on Wikipedia sections test set.


### Sentence Embeddings using XLNet
Currently the XLNet model is under development. It produces Sentence Transformer Models that perform a little bit worse than the BERT models, hence, we not yet release pre-trained models for XLNet.

As soon we have fine-tuned the hyperparameters of XLNet to generate well working sentence embeddings, new pre-trained models will be released.


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
We implemented various loss-functions, that allow training of sentence embeddings from various datasets. These loss-functions are in the package `sentence_transformers.losses`.
 
- *SoftmaxLoss*: Given the sentence embeddings of two sentences, trains a softmax-classifier. Useful for training on datasets like NLI.
- *CosineSimilarityLoss*: Given a sentence pair and a gold similarity score (either between -1 and -1 or between 0 and 1), computes the cosine-similarity between the sentence embeddings and minimizes the mean squared error loss.
- *TripletLoss*: Given a triplet (anchor, positive example, negative example), minimizes the [triplet loss](https://en.wikipedia.org/wiki/Triplet_loss).
- *BatchHardTripletLoss*: Implements the *batch hard triplet loss* from the paper [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737). Each batch must contain multiple examples from the same class. The loss optimizes then the distance between the most-distance positive pair and the closest negative-pair.
- *MultipleNegativesRankingLoss*: Each batch has one positive pair, all other pairs are treated as negative examples. The loss was used in the papers [Efficient Natural Language Response Suggestionfor Smart Reply](https://arxiv.org/pdf/1705.00652.pdf) and [Learning Cross-Lingual Sentence Representations via a Multi-task Dual-Encoder Model](https://arxiv.org/pdf/1810.12836.pdf)

## Models
This framework implements various modules, that can be used sequentially to map a sentence to a sentence embedding. The different modules can be found in the package `sentence_transformers.models`. Each pipeline consists of the following modules.


**Word Embeddings:** These models map tokens to token embeddings.
- **[BERT](sentence_transformers/models/BERT.py)**: Uses pytorch-transformers BERT model to map tokens to vectors. Example:  [examples/training_nli.py](examples/training_nli.py)
- **[XLNet](sentence_transformers/models/XLNet.py)**: Uses pytorch-transformers XLNet model to map tokens to vectors. Example: [examples/training_stsbenchmark_xlnet.py](examples/training_stsbenchmark_xlnet.py)
- **[WordEmbeddings](sentence_transformers/models/WordEmbeddings.py)**: Uses traditional word embeddings like word2vec or GloVe to map tokens to vectors. Example: [examples/training_stsbenchmark_avg_word_embeddings.py](examples/training_stsbenchmark_avg_word_embeddings.py)

**Embedding Transformations:** These model transform token embeddings in some way
- **[LSTM](sentence_transformers/models/LSTM.py)**: Runs a bidirectional LSTM. Example: [examples/training_stsbenchmark_bilstm.py](examples/training_stsbenchmark_bilstm.py).
- **[CNN](sentence_transformers/models/CNN.py)**: Runs a CNN model with multiple kernel sizes. Example: [examples/training_stsbenchmark_cnn.py](examples/training_stsbenchmark_cnn.py).
- **[WordWeights](sentence_transformers/models/WordWeights.py)**: This model can be used after WordEmbeddings and before Pooling to apply a weighting to the token embeddings, for example, a tf-idf weighting. Example: [examples/training_stsbenchmark_tf-idf_word_embeddings.py](examples/training_stsbenchmark_tf-idf_word_embeddings.py).
- **[Pooling](sentence_transformers/models/Pooling.py)**: After tokens are mapped to embeddings, we apply the pooling, were you can compute a mean/max-pooling or use the CLS-token embedding (for BERT and XLNet). You can also combine multiple poolings.

**Sentence Embeddings Models:** These models map a sentence directly to a fixed size sentence embedding:
- **[BoW](sentence_transformers/models/BoW.py)**: Computes a fixed size bag-of-words (BoW) representation of the input text. Can be initialized with IDF-values to create a tf-idf vector. Note, that this model is not trainable. Example


**Sentence Embeddings Transformations:** These models can be added once we have a fixed size sentence embedding.
- **[Dense](sentence_transformers/models/Pooling.py)**: A fully-connected feed-forward network to create a Deep Averging Network (DAN). You can stack multiple Dense models. Example: [examples/training_stsbenchmark_avg_word_embeddings.py](examples/training_stsbenchmark_avg_word_embeddings.py)



## Multitask Training
This code allows multi-task learning with training data from different datasets and with different loss-functions. More documentation will follow soon.


## Application Examples
We present some examples, how the generated sentence embeddings can be used for downstream applications.

### Semantic Search
Semantic search is the task of finding similar sentences to a given sentence. See [examples/application_semantic_search.py](examples/application_semantic_search.py).

We first generate an embedding for all sentences in a corpus:
```
embedder = SentenceTransformer('bert-base-nli-mean-tokens')

# Corpus with example sentences
corpus = ['A man is eating a food.',
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
```
queries = ['A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.', 'A cheetah chases prey on across a field.']
query_embeddings = embedder.encode(queries)
```

We then use scipy to find the most-similar embeddings for queries in the corpus:
```
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
[examples/application_clustering.py](examples/application_clustering.py) depicts an example to cluster similar sentences based on their sentence embedding similarity.

As before, we first compute an embedding for each sentence:
```
embedder = SentenceTransformer('bert-base-nli-mean-tokens')

# Corpus with example sentences
corpus = ['A man is eating a food.',
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

Then, we perform k-means clustering using scipy:
```
# Perform kmean clustering
num_clusters = 5
whitened_corpus = scipy.cluster.vq.whiten(corpus_embeddings)
code_book, _ = scipy.cluster.vq.kmeans(whitened_corpus, num_clusters)
cluster_assignment, _ = scipy.cluster.vq.vq(whitened_corpus, code_book)
```

The output looks like this:
```
Cluster  1
['A man is riding a horse.', 'A man is riding a white horse on an enclosed ground.']

Cluster  2
['A man is eating a food.', 'A man is eating a piece of bread.', 'A man is eating pasta.']

Cluster  3
['A monkey is playing drums.', 'Someone in a gorilla costume is playing a set of drums.']

Cluster  4
['The girl is carrying a baby.', 'The baby is carried by the woman']

Cluster  5
['A cheetah is running behind its prey.', 'A cheetah chases prey on across a field.']
```

## Citing & Authors
A publication describing this framework will be released soon at EMNLP 2019.

The main contributors of this repository are:
- [Nils Reimers](https://github.com/nreimers)
- [Gregor Geigle](https://github.com/aaronsom)

Contact person: Nils Reimers, Rnils@web.de

https://www.ukp.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.







