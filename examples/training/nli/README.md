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

