# Natural Language Inference with Sparse Encoders

Given two sentence (premise and hypothesis), Natural Language Inference (NLI) is the task of deciding if the premise entails the hypothesis, if they are contradiction, or if they are neutral. Commonly used NLI dataset are [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) and [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli). 

[Conneau et al.](https://arxiv.org/abs/1705.02364) showed that NLI data can be quite useful when training Sentence Embedding methods. We also found this in our [Sentence-BERT-Paper](https://arxiv.org/abs/1908.10084) and often use NLI as a first fine-tuning step for sparse encoder methods.

To train on NLI, see the following example file:
**[train_splade_nli.py](train_splade_nli.py)**:
    ```{eval-rst}
    This script trains a `SparseEncoder` (specifically a SPLADE-like model, fine-tuning e.g., `naver/splade-cocondenser-ensembledistil`) for NLI. It uses the :class:`~sentence_transformers.sparse_encoder.losses.SpladeLoss`. This loss combines a ranking loss (like :class:`~sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss`) with regularization terms to encourage sparsity in the learned term weights, characteristic of SPLADE models. The script uses the AllNLI dataset with "pair-score" configuration, likely treating entailment pairs as positives for the ranking component. This approach aims to create highly sparse and effective representations for retrieval.
    ```


## Data
We combine [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) and [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) into a dataset we call [AllNLI](https://huggingface.co/datasets/sentence-transformers/all-nli). These two datasets contain sentence pairs and one of three labels: entailment, neutral, contradiction:

| Sentence A (Premise) | Sentence B (Hypothesis) | Label |
| --- | --- | --- |
| A soccer game with multiple males playing. | Some men are playing a sport. | entailment |
| An older and younger man smiling. | Two men are smiling and laughing at the cats playing on the floor. | neutral |
| A man inspects the uniform of a figure in some East Asian country. | The man is sleeping. | contradiction |

We format AllNLI in a few different subsets, compatible with different loss functions. See for example the [triplet subset of AllNLI](https://huggingface.co/datasets/sentence-transformers/all-nli/viewer/triplet).


## SparseMultipleNegativesRankingLoss
```{eval-rst}
The :class:`~sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss` produces great sparse sentence representations for this type of trainings.
```

The training data for SparseMultipleNegativesRankingLoss consists of sentence pairs [(a<sub>1</sub>, b<sub>1</sub>), ..., (a<sub>n</sub>, b<sub>n</sub>)] where we assume that (a<sub>i</sub>, b<sub>i</sub>) are similar sentences and (a<sub>i</sub>, b<sub>j</sub>) are dissimilar sentences for i != j. The minimizes the distance between (a<sub>i</sub>, b<sub>i</sub>) while it simultaneously maximizes the distance (a<sub>i</sub>, b<sub>j</sub>) for all i != j. For example, in the following picture:

<img src="https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/MultipleNegativeRankingLoss.png" alt="SBERT MultipleNegativeRankingLoss" width="350"/>

The distance between (a<sub>1</sub>, b<sub>1</sub>) is reduced, while the distance between (a<sub>1</sub>, b<sub>2...5</sub>) will be increased. The same is done for a<sub>2</sub>, ..., a<sub>5</sub>.

```{eval-rst}
Using :class:`~sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss` with NLI is rather easy: We define sentences that have an *entailment* label as positive pairs. E.g, we have pairs like (*"A soccer game with multiple males playing."*, *"Some men are playing a sport."*) and want that these pairs are close in vector space. The `pair subset of AllNLI <https://huggingface.co/datasets/sentence-transformers/all-nli/viewer/pair>`_ has been prepared in this format.
```

### SparseMultipleNegativesRankingLoss with Hard Negatives

We can further improve SparseMultipleNegativesRankingLoss by providing triplets rather than pairs: [(a<sub>1</sub>, b<sub>1</sub>, c<sub>1</sub>), ..., (a<sub>n</sub>, b<sub>n</sub>, c<sub>n</sub>)]. The samples for c<sub>i</sub> are so-called hard-negatives: On a lexical level, they are similar to a<sub>i</sub> and b<sub>i</sub>, but on a semantic level, they mean different things and should not be close to a<sub>i</sub> in the vector space.

For NLI data, we can use the contradiction-label to create such triplets with a hard negative. So our triplets look like this:
("*A soccer game with multiple males playing."*, *"Some men are playing a sport."*, *"A group of men playing a baseball game."*). We want the sentences *"A soccer game with multiple males playing."* and *"Some men are playing a sport."* to be close in the vector space, while there should be a larger distance between *"A soccer game with multiple males playing."* and "*A group of men playing a baseball game."*. The [triplet subset of AllNLI](https://huggingface.co/datasets/sentence-transformers/all-nli/viewer/triplet) has been prepared in this format.
