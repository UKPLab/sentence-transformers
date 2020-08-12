# Pretrained Models

We provide various pre-trained models. Using these models is easy:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('model_name')
```

Alternatively, you can download and unzip them from [here](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/).

## Choosing the right Pretrained Model
Sadly there cannot exist a universal model that performs great on all possible tasks. Models strong on one task, will be weak for another task. Hence, it is important to select the right model for your task.


## Semantic Textual Similarity
The following models were optimized for [Semantic Textual Similarity](usage/semantic_textual_similarity) (STS): You can compute the embeddings for two sentences and then use cosine-similarity to get a score -1 ... 1 to indicate their semantic similarity. They were trained on [NLI](pretrained-models/nli-models.md) and [STS](pretrained-models/sts-models.md) data. They are evaluated on the STSbenchmark dataset.

We can recommend this models as general purpose models. The best available models are:
- **roberta-large-nli-stsb-mean-tokens** - STSb performance: 86.39
- **roberta-base-nli-stsb-mean-tokens** - STSb performance: 85.44
- **bert-large-nli-stsb-mean-tokens** - STSb performance: 85.29
- **distilbert-base-nli-stsb-mean-tokens** - STSb performance:  85.16

[» Full List of STS Models](https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0)

I can recommend the **distilbert-base-nli-stsb-mean-tokens** model, which gives a nice balance between speed and performance.

## Duplicate Questions Detection

The following models were trained duplicate questions mining and duplicate questions retrieval. You can use them to detect (see [paraphrase mining](usage/paraphrase_mining) ) or search for similar questions (see [semantic search](usage/semantic_search)). 

Available models:
- **distilbert-base-nli-stsb-quora-ranking**


## Wikipedia Sections
The following models is trained on the dataset from Dor et al. 2018, [Learning Thematic Similarity Metric Using Triplet Networks](https://aclweb.org/anthology/P18-2009) and learns if two sentences belong to the same section in a Wikipedia page or not. It can be used to do fine-grained clustering of similar sentences into sections / topics. [Further details](pretrained-models/wikipedia-sections-models.md)    

- **bert-base-wikipedia-sections-mean-tokens**: 80.42% accuracy on Wikipedia Triplets test set.