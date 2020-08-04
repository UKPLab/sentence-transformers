# Duplicate Questions Information Retrieval

This folder contains scripts that demonstrate how to train SentenceTransformers for **Information Retrieval**. As simple example, we will use the [Quora Duplicate Questions dataset](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs). It contains over 500,000 sentences with over 400,000 pairwise annotation whether two questions are a duplicate or not.

## Pretrained Model


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

### Information Retrieval


## Training
