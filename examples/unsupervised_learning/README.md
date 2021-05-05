# Overview

This page contains a collection of unsupervised learning methods to learn sentence embeddings. The methods have in common that they **do not require labeled training data**. Instead, they can learn semantically meaningful sentence embeddings just from the text itself.

**Note:** Compared to supervised approaches with in-domain training data, unsupervised approaches  achieve low performances, especially for challenging tasks. But for many tasks training data is not available and would be expensive to create. 

## Sentence-Based Approaches

The following approaches only require sentences from your target domain.

**Disclaimer:** Unsupervised Sentence Embedding methods are still an active research area and the results are not well predictable. In many cases, pre-trained models that use labeled data work better also for other (new) domains. 

### TSDAE
In our work [TSDAE (Tranformer-based Denoising AutoEncoder)](https://arxiv.org/abs/2104.06979) we present an unsupervised sentence embedding learning method based on denoising auto-encoders:

![](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/TSDAE.png)

We add noise to the input text, in our case, we delete about 60% of the words in the text. The encoder maps this input to a fixed-sized sentence embeddings. A decoder then tries to re-create the original text without the noise. Later, we use the encoder as the sentence embedding methods.

See **[TSDAE](tsdae/README.md)** for more information and training examples.

### SimCSE

Gao et al. present in [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821) a method that passes the same sentence twice to the sentence embedding encoder. Due to the drop-out, it will be encoded at slightly different positions in vector space. 

The distance between these two embeddings will be minized, while the distance to other embeddings of the other sentences in the same batch will be maximized.

![SimCSE working](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/SimCSE.png)

See **[SimCSE](SimCSE/README.md)** for more information and training examples.

### CT

Carlsson et al. present in [Semantic Re-Tuning With Contrastive Tension (CT)](https://openreview.net/pdf?id=Ov_sMNau-PF) an unsupervised method that uses two models: If the same sentences are passed to Model1 and Model2, then the respective sentence embeddings should get a large dot-score. If the different sentences are passed, then the sentence embeddings should get a low score.

![CT working](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/CT.jpg)

See **[CT](CT/README.md)** for more information and training examples.

### CT (In-Batch Negative Sampling)

The CT method from Carlsson et al. provides sentence pairs to the two models. This can be improved by using in-batch negative sampling: Model1 and Model2 both encode the same set of sentences. We maximize the scores for matching indexes (i.e. Model1(S_i) and Model2(S_i)) while we minimize the scores for different indexes (i.e. Model1(S_i) and Model2(S_j) for i != j).

See **[CT_In-Batch_Negatives](CT_In-Batch_Negatives/README.md)** for more information and training examples.

### Performance Comparison
Currently we conduct experiments which unsupervised sentence embedding methods yields the best results. In terms of run-time are SimCSE and CT-Improved quite fast to train, while TSDAE takes the longest to train.

---------------------

## Pre-Training

Pre-training methods are used to first adapt the transformer model to your domain on a large (unlabeled) corpus. Then, fine-tuning is used with labeled data. In our [TSDAE-paper](https://arxiv.org/abs/2104.06979) we showed the impact pre-training can have on the performance of supervised models.

### Masked Language Model (MLM)
BERT showed that Masked Language Model (MLM) is a powerful pre-training approach. It is advisable to first run MLM a large dataset from your domain before you do fine-tuning. See **[MLM](MLM/README.md)** for more information and training examples.


------------------------

## GenQ

In our paper [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)  we present a method to learn a semantic search method by generating queries for given passages.

We pass all passages in our collection through a trained T5 model, which generates potential queries from users. We then use these (query, passage) pairs to train a SentenceTransformer model.

![Query Generation](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/query-generation.png)

See **[GenQ](query_generation/README.md)** for more information and training examples.