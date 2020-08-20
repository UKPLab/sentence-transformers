# Training Overview

Each task is unique, and having sentence / text embeddings tuned for that specific task greatly improves the performance.

SentenceTransformers was designed in such way that fine-tuning your own sentence / text embeddings models is easy. It provides most of the building blocks that you can stick together to tune embeddings for your specific task.

Sadly there is no single training strategy that works for all use-cases. Instead, which training strategy  to use greatly depends on your available data and on your target task.

In the **Training** section, I will discuss the fundamentals of training your own embedding models with SentenceTransformers. In the **Training Examples** section, I will provide examples how to tune embedding models for common real-world applications.

## Network Architecture

For sentence / text embeddings, we want to map a variable length input text to a fixed sized dense vector. The most basic network architecture we can use is the following:

![SBERT  Network Architecture](../img/SBERT_Architecture.png "SBERT Siamese Architecture")


We feed the input sentence or text into a transformer network like BERT. BERT produces contextualized word embeddings for all input tokens in our text. As we want a fixed-sized output representation (vector u), we need a pooling layer. Different pooling options are available, the most basic one is mean-pooling: We simply average all contextualized word embeddings BERT is giving us. This gives us a fixed 768 dimensional output vector independet how long our input text was.

The depicted architecture, consisting on a BERT layer and a pooling layer is one final SentenceTransformer model.

## Creating Networks from Scratch
 
 In the quick start & usage examples, we used pre-trained SentenceTransformer models that already come with a BERT layer and a pooling layer.
 
 But we can create the networks architectures from scratch by defining the individual layers. For example, the following code would create the depicted network architecture:
 
```python
from sentence_transformers import SentenceTransformer, models

word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
```

First we define our individual layers, in this case, we define 'bert-base-uncased' as the *word_embedding_model*. We limit that layer to a maximal sequence length of 256, texts longer than that will be truncated. Further, we create a (mean) pooling layer. We create a new *SentenceTransformer* model by calling `SentenceTransformer(modules=[word_embedding_model, pooling_model])`. For the *modules* parameter, we pass a list of layers which are executed consecutively. Input text are first passed to the first entry (*word_embedding_model*). The output is then passed to the second entry (*pooling_model*), which then returns our sentence embedding.

We can also construct more complex models:
```python
from sentence_transformers import SentenceTransformer, models
from torch import nn

word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
```

Here, we add a on top of the pooling layer a fully connected dense layer with Tanh activation, which performs a down-project to 256 dimensions. Hence, embeddings by this model will only have 256 instead of 768 dimensions.

For all available building blocks see [» Models Package Reference](../package_reference/models.md)

## Training Data 
 
 To represent our training data, we use the `InputExample` class to store training examples. As parameters, it accepts texts, which is a list of strings representing our pairs (or triplets). Further, we can also pass a label (either float or int). The following shows a simple example, where we pass text pairs to `InputExample` together with a label indicating the semantic similarity.
 
 ```python
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample
from torch.utils.data import DataLoader

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
    InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
 ```

To prepare the examples for training, we provide a custom `SentencesDataset`, which is a [custom PyTorch dataset](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html). It accepts as parameters the list with `InputExamples` and the `SentenceTransformer` model.

We can wrap `SentencesDataset` with the standard PyTorch `DataLoader`, which produces for example batches and allows us to shuffle the data for training.

## Data Readers

For common dataset formats, we provide `DataReaders` that read text (or .gz-files), and generate according lists with `InputExample`.  This are just help function and not required to train a model.

Assume you have a tab-seperated file with the following format:
```
My first sentence   The second sentence   3.5
```


The following script allows to read this file. You specifiy the delimiter between the columns, the columns for sentence1 (s1), sentence2 (s2), and the score. Further, the semantic similarity might be a score between 0 and 5. However, for training, we must have scores between 0 and 1. Hence, we tell the reader to normalize the score and that the lowest score is 0 and the highest 5.

```python
from sentence_transformers import readers
import csv

sts_reader = readers.STSDataReader('path/to/folder/with/files', s1_col_idx=0, s2_col_idx=1, score_col_idx=2, delimiter="\t",
                 quoting=csv.QUOTE_NONE, normalize_scores=True, min_score=0, max_score=5)

train_examples = sts_reader.get_examples('train.tsv')
```

This gives back a list of `InputExample` that has as `texts`-parameter the two sentences and as `label` the similarity score (0...1).

To see all readers, see [Readers](../package_reference/readers).
 
## Loss Functions

The loss function plays a critical role when fine-tuning the model. It determines how well our embedding model will work for the specific downstream task.

Sadly there is no "one size fits all" loss function. Which loss function is suitable depends on the available training data and on the target task.


To fine-tune our network, we need somehow to tell our network which sentence pairs are similar, and should be close in vector space, and which pairs are dissimilar, and should be far away in vector space.

The most simple way is to have sentence pairs annotated with a score indicating their similarity, e.g. on a scale 0 to 1. We can then train the network with a Siamese Network Architecture (for details see: [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084))

![SBERT Siamese Network Architecture](../img/SBERT_Siamese_Network.png "SBERT Siamese Architecture")


For each sentence pair, we pass sentence A and sentence B through our network which yields the embeddings *u* und *v*. The similarity of these embeddings is computed using cosine similarity and the result is compared to the gold similarity score. This allows our network to be fine-tuned and to recognize the similarity of sentences.


A minimal example with `CosineSimilarityLoss` is the following:
```python
 ```python
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
from torch.utils.data import DataLoader

#Define the model. Either from scratch of by loading a pre-trained model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

#Define your train examples. You need more than just two examples...
train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
    InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]

#Define your train dataset, the dataloader and the train loss
train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
```


We tune the model by calling model.fit(). We pass a list of `train_objectives`, which constist of tuples `(dataloader, loss_function)`. We can pass more than one tuple in order to perform multi-task learning on several datasets with different loss functions.

The `fit` method accepts the following parameter:

```eval_rst
.. autoclass:: sentence_transformers.SentenceTransformer
    :members: fit
```

## Evaluators

During training, we usually want to measure the performance to see if the performance improves. For this, the *[sentence_transformers.evaluation](../package_reference/evaluation)* package exists. It contains various evaluators which we can pass to the `fit`-method. These evaluators are run periodically during training. Further, they return a score and only the model with the highest score will be stored on disc.

The usage is simple:
```python
from sentence_transformers import evaluation
sentences1 = ['This list contains the first column', 'With your sentences', 'You want your model to evaluate on']
sentences2 = ['Sentences contains the other column', 'The evaluator matches sentences1[i] with sentences2[i]', 'Compute the cosine similarity and compares it to scores[i]']
scores = [0.3, 0.6, 0.2]

evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

# ... Your other code to load training data

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, evaluator=evaluator, evaluation_steps=500)
```

