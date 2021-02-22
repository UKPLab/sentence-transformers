# Synthetic Query Generation
This folder shows an example, how we can train an [asymmetric semantic search](../../applications/semantic-search/) without requiring training data.

## Background
In [asymmetric semantic search](../../applications/semantic-search/), the user provides a (short) query like some keywords or a question. We then want to retrieve a longer text passage that provides the answer.

For example:
```
query: What is Python?
passage to retrieve: Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects.
```

We showed how to train such models if sufficient training data (query & relevant passage) is available here: [Training MS MARCO dataset](../../training/ms_marco) 

In this tutorial, we show to train such models if  **no training data is available**, i.e., if you don't have thousands of labeled query & relevant passage pairs.

## Overview

We use **synthethic query generation** to achieve our gold. We start with the passage from our document collection and create for these possible queries users might ask / might search for.

![BiEncoder](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/query_generation.png)


For example, we have the following text passage:
```
 Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects.
```

We pass this passage through a specially trained [T5 model](https://arxiv.org/abs/1910.10683) which generates possible queries for us. For the above passage, it might generate these queries:
- What is python
- definition python
- what language uses whitespaces


We then use these generated queries to create our training set:
```
(What is python, Python is an interpreted...)
(definition python, Python is an interpreted...)
(what language uses whitespaces, Python is an interpreted...)
````

And train our SentenceTransformer bi-encoder with it.

## Query Generation

https://github.com/castorini/docTTTTTquery

https://huggingface.co/blog/how-to-generate

## Bi-Encoder Training

## Performance