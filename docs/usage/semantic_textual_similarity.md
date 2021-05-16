# Semantic Textual Similarity

Once you have  [sentence embeddings computed](../../examples/applications/computing-embeddings/README.md), you usually want to compare them to each other. Here, I show you how you can compute the cosine similarity between embeddings, for example, to measure the semantic similarity of two texts.

```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

# Two lists of sentences
sentences1 = ['The cat sits outside',
             'A man is playing guitar',
             'The new movie is awesome']

sentences2 = ['The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great']

#Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

#Compute cosine-similarits
cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

#Output the pairs with their score
for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
```

We pass the `convert_to_tensor=True` parameter to the encode function. This will return a pytorch tensor containing our embeddings. We can then call `util.pytorch_cos_sim(A, B)` which computes the cosine similarity between all vectors in *A* and all vectors in *B*. 

It returns in the above example a 3x3 matrix with the respective cosine similarity scores for all possible pairs between *embeddings1* and *embeddings2*.


You can use this function also to find out the pairs with the highest cosine similarity scores:
```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-distilroberta-base-v1')

# Single list of sentences
sentences = ['The cat sits outside',
             'A man is playing guitar',
             'I love pasta',
             'The new movie is awesome',
             'The cat plays in the garden',
             'A woman watches TV',
             'The new movie is so great',
             'Do you like pizza?']

#Compute embeddings
embeddings = model.encode(sentences, convert_to_tensor=True)

#Compute cosine-similarities for each sentence with each other sentence
cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)

#Find the pairs with the highest cosine similarity scores
pairs = []
for i in range(len(cosine_scores)-1):
    for j in range(i+1, len(cosine_scores)):
        pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

#Sort scores in decreasing order
pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

for pair in pairs[0:10]:
    i, j = pair['index']
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], pair['score']))
```

Note, in the above approach we use a brute-force approach to find the highest scoring pairs, which has a quadratic complexity. For long lists of sentences, this might be infeasible. If you want find the highest scoring pairs in a long list of sentences, have a look at [Paraphrase Mining](../../examples/applications/paraphrase-mining/README.md).