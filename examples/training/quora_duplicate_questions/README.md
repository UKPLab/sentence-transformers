# Quora Duplicate Questions

This folder contains scripts that demonstrate how to train SentenceTransformers for **Information Retrieval**. As simple example, we will use the [Quora Duplicate Questions dataset](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs). It contains over 500,000 sentences with over 400,000 pairwise annotations whether two questions are a duplicate or not.

## Pretrained Models

Currently the following models trained on Quora Duplicate Questions are available:
* **distilbert-base-nli-stsb-quora-ranking**:  We extended the *distilbert-base-nli-stsb-mean-tokens* model and trained it with *OnlineContrastiveLoss* and with *MultipleNegativesRankingLoss* on the Quora Duplicate questions dataset. For the code, see [training_multi-task-learning.py](training_multi-task-learning.py)
* **distilbert-multilingual-nli-stsb-quora-ranking**: Extension of *distilbert-base-nli-stsb-quora-ranking* to be multi-lingual. Trained on parallel data for 50 languages.

You can load & use pre-trained models like this:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('model_name')
```


## Dataset
As dataset to train a **Duplicate Questions Semantic Search Engine** we use [Quora Duplicate Questions dataset](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs). The original format looks like this:
```
id	qid1	qid2	question1	question2	is_duplicate
0	1	2	What is the step by step guide to invest in share market in india?	What is the step by step guide to invest in share market?	0
1	3	4	What is the story of Kohinoor (Koh-i-Noor) Diamond?	What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?	0
```

As a first step, we process this file to create distinct train/dev/test splits for different tasks. We define the following tasks:
- **Duplicate Questions Classification**: Given two questions, are these questions duplicates? This is the original task as defined by Quora, however, it is rather a unpractical task. How do we retrieve possible duplicates in a large corpus for a given question? Further, models performing well on this classification task do not necessarily perform well on the following two task.
- **Duplicate Questions Mining**: Given a large set (like 100k) of questions, identify all question pairs that are duplicates.
- **Duplicate Questions Information Retrieval**: Given a large corpus (350k+) of questions. For a new, unseen question, find the most related (i.e. duplicate) questions in this corpus.


**Download**: You can download the finished dataset here: [quora-IR-dataset.zip](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/quora-IR-dataset.zip)

For details on the creation of the dataset, see [create_splits.py](create_splits.py).


## Usage

### Duplicate Questions Mining

Given a large set of sentences (in this case questions), identify all pairs that are duplicates. See [Paraphrase Mining](../../applications/paraphrase-mining/README.md) for an example how to use sentence transformers to mine for duplicate questions / paraphrases. This approach can be scaled to hundred thousands of sentences given you have enough memory.

### Semantic Search

The model can also be used for Information Retrieval / Semantic Search. Given a new question, search a large corpus of hundred thousands of questions for duplicate questions. Given you have enough memory, this approach works well to copora up in the Millions (depending on your real-time requirements).

For an interactive example, see [Semantic Search](../../applications/semantic-search/README.md).


## Training

Choosing the right loss function is crucial for getting well working sentence embeddings. For the given task, we two loss functions are especially suitable: **ConstrativeLoss** and **MultipleNegativesRankingLoss**

### Constrative Loss
For the complete example, see [training_OnlineContrastiveLoss.py](training_OnlineContrastiveLoss.py).

In the original dataset, we have questions given with a label of 0=not duplicate and 1=duplicate. In that case, we can use constrative loss: Similar pairs with label 1 are pulled together, so that they are close in vector space. Dissimilar pairs, that are closer than a defined margin, are pushed away in vector space.

Choosing the distance function and especially choosing a sensible margin are quite important for the success of constrative loss. In the given example, we use cosine_distance (which is 1-cosine_similarity) with a margin of 0.5. I.e., non-duplicate questions should have a cosine_distance of at least 0.5 (which is equivalent to a 0.5 cosine similarity difference).

An improved version of constrative loss is OnlineConstrativeLoss, which looks which negative pairs have a lower distance that the largest positive pair and which positive pairs have a higher distance than the lowest distance of negative pairs. I.e., this loss automatically detects the hard cases in a batch and computes the loss only for these cases.

The loss can be used like this:
```python
train_samples = []
with open(os.path.join(dataset_path, "classification/train_pairs.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        sample = InputExample(texts=[row['question1'], row['question2']], label=int(row['is_duplicate']))
        train_samples.append(sample)


train_dataset = SentencesDataset(train_samples, model=model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.OnlineContrastiveLoss(model=model, distance_metric=distance_metric, margin=margin)
``` 

For each row in our train dataset, we create new InputExample objects and the two questions as texts and the is_duplicate as the label.



## MultipleNegativesRankingLoss
For the complete example, see [training_MultipleNegativesRankingLoss.py](training_MultipleNegativesRankingLoss.py).

*MultipleNegativesRankingLoss* is especially suitable for Information Retrieval / Semantic Search. A nice advantage of *MultipleNegativesRankingLoss* is that it only requires positive pairs, i.e., we only need examples of duplicate questions.

From all pairs, we sample a mini-batch *(a_1, b_1), ..., (a_n, b_n)* where *(a_i, b_i)* is a duplicate question.

MultipleNegativesRankingLoss now uses all *b_j* with j != i as negative example for *(a_i, b_i)*. For example, for *a_1* we have given the options *(b_1, ..., b_n)* and we need to identify which is the correct duplicate question to *a_1*. We do this by computing the dot-product between the embedding of *a_1* and all *b*'s and softmax normalize it so that we get a proability distribution over *(b_1, ..., b_n)*. In the best case, the positive example *b_1* get a probability of close to 1 while all others get scores close to 0. We use negative log-likelihood to compute the loss.


*MultipleNegativesRankingLoss* implements this idea in an efficient way so that the embeddings are re-used. With a batch-size of 64, we have 64 positive pairs and each positive pairs has 64-1 negative distractors. 


Using the loss is easy and does not require tuning of any hyperparameters:
```python
train_samples = []
with open(os.path.join(dataset_path, "classification/train_pairs.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['is_duplicate'] == '1':
            train_samples.append(InputExample(texts=[row['question1'], row['question2']], label=1))
            train_samples.append(InputExample(texts=[row['question2'], row['question1']], label=1)) #if A is a duplicate of B, then B is a duplicate of A


# After reading the train_samples, we create a SentencesDataset and a DataLoader
train_dataset = SentencesDataset(train_samples, model=model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model)
```

We only use the positive examples. As 'is_duplicate' is a symmetric relation, we not only add (A, B) but also (B, A) to our training sample set.

**Note 1:** Increasing the batch sizes usually yields better results, as the task gets harder. It is more difficult to identify the correct duplicate question out of a set of 100 questions than out of a set of only 10 questions. So it is advisable to set the training batch size as large as possible. I trained it with a batch size of 350 on 32 GB GPU memory.

**Note 2:** MultipleNegativesRankingLoss only works if *(a_i, b_j)* with j != i is actually a negative, non-duplicate question pair. In few instances, this assumption is wrong. But in the majority of cases, if we sample two random questions, they are not duplicates. If your dataset cannot fullfil this property,  MultipleNegativesRankingLoss might not work well.

### Multi-Task-Learning
Constrative Loss works well for pair classification, i.e., given two pairs, are these duplicates or not. It pushes negative pairs far away in vector space, so that the distinguishing between duplicate and non-duplicate pairs works good.

MultipleNegativesRankingLoss on the other sides mainly reduces the distance between positive pairs out of large set of possible candidates. However, the distance between  non-duplicate questions is not so large, so that this loss does not work that weill for pair classification.

In [training_multi-task-learning.py](training_multi-task-learning.py) I demonstrate how we can train the network with both losses. The essential code is to define both losses and to pass it to the fit method.
```python
train_samples_MultipleNegativesRankingLoss = []
train_samples_ConstrativeLoss = []

with open(os.path.join(dataset_path, "classification/train_pairs.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        train_samples_ConstrativeLoss.append(InputExample(texts=[row['question1'], row['question2']], label=int(row['is_duplicate'])))
        if row['is_duplicate'] == '1':
            train_samples_MultipleNegativesRankingLoss.append(InputExample(texts=[row['question1'], row['question2']], label=1))
            train_samples_MultipleNegativesRankingLoss.append(InputExample(texts=[row['question2'], row['question1']], label=1))  # if A is a duplicate of B, then B is a duplicate of A

# Create data loader and loss for MultipleNegativesRankingLoss
train_dataset_MultipleNegativesRankingLoss = SentencesDataset(train_samples_MultipleNegativesRankingLoss, model=model)
train_dataloader_MultipleNegativesRankingLoss = DataLoader(train_dataset_MultipleNegativesRankingLoss, shuffle=True, batch_size=train_batch_size)
train_loss_MultipleNegativesRankingLoss = losses.MultipleNegativesRankingLoss(model)


# Create data loader and loss for OnlineContrastiveLoss
train_dataset_ConstrativeLoss = SentencesDataset(train_samples_ConstrativeLoss, model=model)
train_dataloader_ConstrativeLoss = DataLoader(train_dataset_ConstrativeLoss, shuffle=True, batch_size=train_batch_size)
train_loss_ConstrativeLoss = losses.OnlineContrastiveLoss(model=model, distance_metric=distance_metric, margin=margin)

# .....
# Train the model
model.fit(train_objectives=[(train_dataloader_MultipleNegativesRankingLoss, train_loss_MultipleNegativesRankingLoss), (train_dataloader_ConstrativeLoss, train_loss_ConstrativeLoss)],
          evaluator=seq_evaluator,
          epochs=num_epochs,
          warmup_steps=1000,
          output_path=model_save_path
          )
```

