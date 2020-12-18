# Clustering
Sentence-Transformers can be used in different ways to perform clustering of small or large set of sentences.

## k-Means
[kmeans.py](kmeans.py) contains an example of using [K-means Clustering Algorithm](https://scikit-learn.org/stable/modules/clustering.html#k-means). K-Means requires that the number of clusters is specified beforehand. The sentences are clustered in groups of about euqal size.
 
## Agglomerative Clustering
[agglomerative.py](agglomerative.py) shows an example of using [Hierarchical clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) using the [Agglomerative Clustering  Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering). In contrast to k-means, we can specify a threshold for the clustering: Clusters below that threshold are merged. This algorithm can be useful if the number of clusters is unknown. By the threshold, we can control if we want to have many small and fine-grained cluster or few coarse-grained clusters.

## Fast Clustering

Agglomerative Clustering is for larger datasets quite slow, so it is only applicable for maybe a few thousand sentences.

In [fast_clustering.py](fast_clustering.py) we present a clustering algorithm that is tuned for large datasets (50k sentences in less than 5 seconds). In a large list of sentences it searches for local communities: A local community is a set of highly similar sentences. 

You can configure the threshold of cosine-similarity for which we consider two sentences as similar. Also, you can specific the minimal size for a local community. This allows you to get either large coarse-grained cluster or small fine-grained clusters. 

We apply it on the [Quora Duplicate Questions](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs) dataset and the output will looks something like this:

```
Cluster 1, #109 Elements
         How do I improve my English speaking?
         How could I improve my English?
         How can I improve my English speaking ability?

Cluster 2, #99 Elements
         Will the decision to demonetize 500 and 1000 rupee notes help to curb black money?
         The decision of Indian Government to demonetize ₹500 and ₹1000 notes? Is Right or wrong?
         What do you think about Modi's new policy on the ban of Rs 500 and Rs 1000 notes?

Cluster 3, #61 Elements
         What are the best way of loose the weight?
         What is the best method of losing weight?
         What are the best simple ways to loose weight?

...

Cluster 21, #25 Elements
         Why is Saltwater Taffy candy imported in Portugal?
         Why is saltwater taffy candy imported in Brazil?
         Why is Saltwater taffy candy imported in Japan?
```


## Topic Modeling
Topic modeling is the process of discovering topics in a collection of documents. 

An example is shown in the following picture, which shows the identified topics in the 20 newsgroup dataset:
![20news](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/20news_semantic.png) 

For each topic, you want to extract the words that describe this topic:
![20news](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/20news_top2vec.png) 

Sentence-Transformers can be used to identify these topics in a collection of sentences, paragraphs or short documents. For

For an excellent tutorial, see [Topic Modeling with BERT](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6) as well as the repositories [Top2Vec](https://github.com/ddangelov/Top2Vec) and [BERTopic](https://github.com/MaartenGr/BERTopic).
 
 
 Image source: [Top2Vec: Distributed Representations of Topics](https://arxiv.org/abs/2008.09470)