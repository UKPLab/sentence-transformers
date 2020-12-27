# Quickstart
Once you have SentenceTransformers [installed](installation.md), the usage is simple:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

#Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']

#Sentences are encoded by calling model.encode()
sentence_embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
```


With `SentenceTransformer('paraphrase-distilroberta-base-v1')` we define which sentence transformer model we like to load. In this example, we load *paraphrase-distilroberta-base-v1*, which is a DistilBERT-base-uncased model fine tuned on a large dataset of paraphrase sentences.

BERT (and other transformer networks) output for each token in our input text an embedding. In order to create a fixed-sized sentence embedding out of this, the model applies mean pooling, i.e., the output embeddings for all tokens are averaged to yield a 768-dimensional vector.

## Comparing Sentence Similarities

The sentences (texts) are mapped such that sentences with similar meanings are close in vector space. One common method to measure the similarity in vector space is to use [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity). For two sentences, this can be done like this:

```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

#Sentences are encoded by calling model.encode()
emb1 = model.encode("This is a red cat with a hat.")
emb2 = model.encode("Have you seen my red cat?")

cos_sim = util.pytorch_cos_sim(emb1, emb2)
print("Cosine-Similarity:", cos_sim)
```

If you have a list with more sentences, you can use the following code example:
```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

sentences = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'Someone in a gorilla costume is playing a set of drums.'
          ]

#Encode all sentences
embeddings = model.encode(sentences)

#Compute cosine similarity between all pairs
cos_sim = util.pytorch_cos_sim(embeddings, embeddings)

#Add all pairs to a list with their cosine similarity score
all_sentence_combinations = []
for i in range(len(cos_sim)-1):
    for j in range(i+1, len(cos_sim)):
        all_sentence_combinations.append([cos_sim[i][j], i, j])

#Sort list by the highest cosine similarity score
all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)

print("Top-5 most similar pairs:")
for score, i, j in all_sentence_combinations[0:5]:
    print("{} \t {} \t {:.4f}".format(sentences[i], sentences[j], cos_sim[i][j]))
```

See on the left the *Usage* sections for more examples how to use SentenceTransformers.

## Pre-Trained Models
Various pre-trained models exists optimized for many tasks exists. For a full list, see **[Pretrained Models](pretrained_models.md)**. 



## Training your own Embeddings

Training your own sentence embeddings models for all type of use-cases is easy and requires often only minimal coding effort. For a comprehensive tutorial, see [Training/Overview](training/overview.md).

You can also extend easily existent sentence embeddings models to **further languages**.  For details, see [Multi-Lingual Training](../examples/training/multilingual/README).
