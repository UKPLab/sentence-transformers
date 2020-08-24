# Semantic Search with ElasticSearch
This page describes how to perform semantic search with [ElasticSearch](https://www.elastic.co/elasticsearch/). Also have a look at [semantic search with Python](semantic_search.md).

This example is mainly based on the tutorial: [Text similarity search with vector fields](https://www.elastic.co/blog/text-similarity-search-with-vectors-in-elasticsearch).

## Requirements
For this tutorial you need to have [ElasticSearch](https://www.elastic.co/elasticsearch/) up and running. If you have never worked with ElasticSearch, I recommend to first get familiar with ElasticSearch.

Further, to access ElasticSearch with Python, you need the [Python Elasticsearch Client](https://elasticsearch-py.readthedocs.io/en/master/).

## Dataset

In this example we will index the [Quora Duplicate Questions dataset](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs), a corpus with over 500k questions from Quora. 

## Create Index
As a first step, you need to define an index for your corpus. In order to store the embeddings for your entries, you must define a *dense_vector* field.

For our example, we define the following index:
```
"mappings": {
  "properties": {
    "question": {
      "type": "text"
    },
    "question_vector": {
      "type": "dense_vector",
      "dims": 768
    }
  }
}
```

We define two fields: A text field that contains the question and a *dense_vector* field with 768 dimensions that contains the respective embedding.
