# Paraphrase Mining

Paraphrase mining is the task of finding pharaphrases (texts with identical / similar meaning) in a large corpus of sentences. In [Semantic Textual Similarity](../../../docs/usage/semantic_textual_similarity.md) we saw a simplified version of finding paraphrases in a list of sentences. The approach presented there used a brute-force approach to score and rank all pairs. 

However, as this has a quadratic runtime, it fails to scale to large (10,000 and more) collections of sentences.

For larger collections, *util* offers the *paraphrase_mining* function that can be used like this:
```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Single list of sentences - Possible tens of thousands of sentences
sentences = ['The cat sits outside',
             'A man is playing guitar',
             'I love pasta',
             'The new movie is awesome',
             'The cat plays in the garden',
             'A woman watches TV',
             'The new movie is so great',
             'Do you like pizza?']

paraphrases = util.paraphrase_mining(model, sentences)

for paraphrase in paraphrases[0:10]:
    score, i, j = paraphrase
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], score))
```

The **paraphrase_mining()**-method accepts the following parameters:
```eval_rst
.. autofunction:: sentence_transformers.util.paraphrase_mining
```

Instead of computing all pairwise cosine scores and ranking all possible, combintations, the approach is a bit more complex (and hence efficient). We chunk our corpus into smaller pieces, which is defined by *query_chunk_size* and *corpus_chunk_size*. For example, if we set *query_chunk_size=1000*, we search paraphrases for 1,000 sentences at a time in the remaining corpus (all other sentences). However, the remaining corpus is also chunked, for example, if we set *query_chunk_size=10000*, we look for paraphrases in 10k sentences at a time.

If we pass a list of 20k sentences, we will chunk it to 20x1000 sentences, and each of the query is compared first against sentences 0-10k and then 10k-20k.

This is done to reduce the memory requirement. Increasing both values improves the speed, but increases also the memory requirement.


The next critical thing is finding the pairs with the highest similarities. Instead of getting and sorting all n^2 pairwise scores, we take for each query only the *top_k* scores. So with *top_k=100*, we find at most 100 paraphrases per sentence per chunk. You can play around with *top_k* to the ensure a certain behaviour.

So for example, with
```python
paraphrases = util.paraphrase_mining(model, sentences, corpus_chunk_size=len(sentences), top_k=1)
```

You will get for each sentence only the one most other relevant sentence. Note, if B is the most similar sentence for A, A must not be the most similar sentence for B. So it can happen that the returned list contains entries like (A, B) and (B, C).

The final relevant parameter is *max_pairs*, which determines the maximum number of paraphrase pairs you like to get returned. If you set it to e.g. *max_pairs=100*, you will not get more than 100 paraphrase pairs returned. Usually, you get fewer pairs returned as the list is cleaned of duplicates, e.g., if it contains (A, B) and (B, A), then only one is returned.
 
