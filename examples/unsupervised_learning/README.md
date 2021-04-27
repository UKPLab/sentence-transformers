# Overview

This page contains a collection of unsupervised learning methods to learn sentence embeddings. The methods have in common that they **do not require labeled training data**. Instead, they can learn semantically meaningful sentence embeddings just from the text itself.

**Note:** Compared to supervised approaches with in-domain training data, unsupervised approaches  achieve low performances, especially for challenging tasks. But for many tasks training data is not available and would be expensive to create. 

## TSDAE
In our work [TSDAE (Tranformer-based Denoising AutoEncoder)](https://arxiv.org/abs/2104.06979) we present an unsupervised sentence embedding learning method based on denoising auto-encoders:

![](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/TSDAE.png)

We add noise to the input text, in our case, we delete about 60% of the words in the text. The encoder maps this input to a fixed-sized sentence embeddings. A decoder then tries to re-create the original text without the noise. Later, we use the encoder as the sentence embedding methods.

See **[TSDAE](tsdae/README.md)** for more information and training examples.

## SimCSE

Gao et al. present in [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821) a method that passes the same sentence twice to the sentence embedding encoder. Due to the drop-out, it will be encoded at slightly different positions in vector space. 

The distance between these two embeddings will be minized, while the distance to other embeddings of the other sentences in the same batch will be maximized.

![SimCSE working](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/SimCSE.png)

See **[SimCSE](SimCSE/README.md)** for more information and training examples.

## GenQ

In our paper [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)  we present a method to learn a semantic search method by generating queries for given passages.

We pass all passages in our collection through a trained T5 model, which generates potential queries from users. We then use these (query, passage) pairs to train a SentenceTransformer model.

![Query Generation](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/query-generation.png)

See **[GenQ](query_generation/README.md)** for more information and training examples.