# Natural Language Inference

Given two sentence (premise and hypothesis), Natural Language Inference (NLI) is the task of deciding if the premise entails the hypothesis, if they are contradiction or if they are neutral. Commonly used NLI dataset are [SNLI](https://arxiv.org/abs/1508.05326) and [MultiNLI](https://arxiv.org/abs/1704.05426). 

[Conneau et al.](https://arxiv.org/abs/1705.02364) showed that NLI data can be quite useful when training Sentence Embedding methods. We also found this in our [Sentence-BERT-Paper](https://arxiv.org/abs/1908.10084) and often use NLI as a first fine-tuning step for sentence embedding methods.

To train on NLI, see the follwing example files:
- **[training_nli.py](training_nli.py)** - This example uses the Softmax-Classification-Loss, as described in the [SBERT-Paper](https://arxiv.org/abs/1908.10084), to learn sentence embeddings.
- **[training_nli_v2.py](training_nli_v2.py)** - The Softmax-Classification-Loss, as used in our original SBERT paper, does not yield optimal performance. A better loss is [MultipleNegativesRankingLoss](https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss), where we provide pairs or triplets. In that example, we provide a triplet of the format: (anchor, entailment_sentence, contradiction_sentence). The NLI data provides such triplets. The MultipleNegativesRankingLoss yields much higher performances and is more intuitive than the Softmax-Classifiation-Loss. We have used this loss to train the paraphrase model in our [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813) paper.

## Data
In our experiments we combine [SNLI](https://arxiv.org/abs/1508.05326) and [MultiNLI](https://arxiv.org/abs/1704.05426), which we call AllNLI. These two datasets contain sentence pairs and one of three labels: entailment, neutral, contradiction:

| Sentence A (Premise) | Sentence B (Hypothesis) | Label |
| --- | --- | --- |
| A soccer game with multiple males playing. | Some men are playing a sport. | entailment |
| An older and younger man smiling. | Two men are smiling and laughing at the cats playing on the floor. | neutral |
| A man inspects the uniform of a figure in some East Asian country. | The man is sleeping. | contradiction |





## SoftmaxLoss
[Conneau et al.](https://arxiv.org/abs/1705.02364) described how a softmax classifier on top of a siamese network can be used to learn meaningful sentence representation. We can achieve this by using the  [losses.SoftmaxLoss](../../../docs/package_reference/losses.html#softmaxloss) package.


The softmax loss looks like this:

![SBERT SoftmaxLoss](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/SBERT_SoftmaxLoss.png "SBERT SoftmaxLoss")

We pass the two sentences through our SentenceTransformer network and get the sentence embeddings *u* and *v*. We then concatenate u, v and |u-v| to form one, long vector. This vector is then passed to a softmax classifier, which predicts our three classes (entailnment, neutral, contradiction).

This setup learns sentence embeddings, that can later be used for wide varity of tasks. 

## MultipleNegativesRankingLoss

That the softmax-loss with NLI data produces (relatively) good sentence embeddings is rather coincidental. The [MultipleNegativesRankingLoss](https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss) is much more intuitive and produces also significantly better sentence representations.

The training data for MultipleNegativesRankingLoss consists of sentence pairs [(a<sub>1</sub>, b<sub>1</sub>), ..., (a<sub>n</sub>, b<sub>n</sub>)] where we assume that (a<sub>i</sub>, b<sub>i</sub>) are similar sentences and (a<sub>i</sub>, b<sub>j</sub>) are dissimilar sentences for i != j. The minimizes the distance between (a<sub>i</sub>, b<sub>j</sub>) while it simultaneously maximizes the distance  (a<sub>i</sub>, b<sub>j</sub>) for all i != j.


For example in the following picture:

![](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/MultipleNegativeRankingLoss.png)

The distance between (a<sub>1</sub>, b<sub>1</sub>) is reduced, while the distance between (a<sub>1</sub>, b<sub>2...5</sub>) will be increased. The same is done for a<sub>2</sub>, ..., a<sub>5</sub>.


Using MultipleNegativeRankingLoss with NLI is rather easy: We define sentences that have an *entailment* label as positive pairs. E.g, we have pairs like (*"A soccer game with multiple males playing."*, *"Some men are playing a sport."*) and want that these pairs are close in vector space.

### MultipleNegativesRankingLoss with Hard Negatives

We can further improve MultipleNegativesRankingLoss by not only providing pairs, but by providing triplets: [(a<sub>1</sub>, b<sub>1</sub>, c<sub>1</sub>), ..., (a<sub>n</sub>, b<sub>n</sub>, c<sub>n</sub>)] 

The entry for c<sub>i</sub> are so-called hard-negatives: On a lexical level, they are similar to a<sub>i</sub> and b<sub>i</sub>. But on a semantic level, they mean different things and should not be close in the vector space.

For NLI data, we can use the contradiction-label to create such triplets with a hard negative. So our triplets look like this:
("*A soccer game with multiple males playing."*, *"Some men are playing a sport."*, *"A group of men playing a baseball game."*).

We want the sentences *"A soccer game with multiple males playing."* and *"Some men are playing a sport."* to be close in the vector space, while there should be a larger distance between *"A soccer game with multiple males playing."* and "*A group of men playing a baseball game."*.
