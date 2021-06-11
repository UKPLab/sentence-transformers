# Paraphrase Data

**This page is currently work-in-progress and will be extended in the future**

In our paper [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813) we showed that paraphrase dataset together with [MultipleNegativesRankingLoss](https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss) is a powerful combination to learn sentence embeddings models.

You can find here: [NLI - MultipleNegativesRankingLoss](https://www.sbert.net/examples/training/nli/README.html#multiplenegativesrankingloss) more information how the loss can be used.

In this folder, we collect different datasets and scripts to train using paraphrase data.

## Datasets

You can find here: [sbert.net/datasets/paraphrases](http://sbert.net/datasets/paraphrases) a list of datasets with paraphrases suitable for training.

| Name | Source | #Sentence-Pairs | STSb-dev |
| --- | --- | :---: | :---: |
| [AllNLI.tsv.gz](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases/AllNLI.tsv.gz) | [SNLI](https://nlp.stanford.edu/projects/snli/) + [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/) | 277,230 | 86.54 |
| [sentence-compression.tsv.gz](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases/sentence-compression.tsv.gz) | [sentence-compression](https://github.com/google-research-datasets/sentence-compression) | 180,000 | 84.36 |
| [SimpleWiki.tsv.gz](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases/SimpleWiki.tsv.gz) | [SimpleWiki](https://cs.pomona.edu/~dkauchak/simplification/) | 102,225 | 84.26 |
| [altlex.tsv.gz](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases/altlex.tsv.gz) | [altlex](https://github.com/chridey/altlex/) | 112,696 | 83.34 |
| [msmarco-triplets.tsv.gz](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases/msmarco-triplets.tsv.gz) | [MS MARCO Passages](https://microsoft.github.io/msmarco/) | 5,028,051 | 83.12 |
| [quora_duplicates.tsv.gz](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases/quora_duplicates.tsv.gz) | [Quora](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs) | 103,663 | 82.55 |
| [coco_captions-with-guid.tsv.gz](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases/coco_captions-with-guid.tsv.gz) | [COCO](https://cocodataset.org/) | 828,395 | 82.25
| [flickr30k_captions-with-guid.tsv.gz](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases/flickr30k_captions-with-guid.tsv.gz) | [Flickr 30k](https://shannon.cs.illinois.edu/DenotationGraph/) | 317,695 | 82.04
| [yahoo_answers_title_question.tsv.gz](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases/yahoo_answers_title_question.tsv.gz) | [Yahoo Answers Dataset](https://www.kaggle.com/soumikrakshit/yahoo-answers-dataset) | 659,896 | 81.19 |
| [S2ORC_citation_pairs.tsv.gz](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases/S2ORC_citation_pairs.tsv.gz) | [Semantic Scholar Open Research Corpus](http://s2-public-api-prod.us-west-2.elasticbeanstalk.com/corpus/) | 52,603,982 | 81.02 |
| [yahoo_answers_title_answer.tsv.gz](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases/yahoo_answerstitle_answer.tsv.gz) | [Yahoo Answers Dataset](https://www.kaggle.com/soumikrakshit/yahoo-answers-dataset)  | 1,198,260 | 80.25 
| [stackexchange_duplicate_questions.tsv.gz](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases/stackexchange_duplicate_questions.tsv.gz) | [Stackexchange](https://stackexchange.com/) | 169,438 | 80.37
| [yahoo_answers_question_answer.tsv.gz](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases/yahoo_answers_question_answer.tsv.gz) | [Yahoo Answers Dataset](https://www.kaggle.com/soumikrakshit/yahoo-answers-dataset)  | 681,164 | 79.88 |
| [wiki-atomic-edits.tsv.gz](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases/wiki-atomic-edits.tsv.gz) | [wiki-atomic-edits](https://github.com/google-research-datasets/wiki-atomic-edits) |   22,980,185  | 79.58
| [wiki-split.tsv.gz](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases/wiki-split.tsv.gz) | [wiki-split](https://github.com/google-research-datasets/wiki-split) | 929,944 | 76.59


See the respective linked source website for the dataset license.


All datasets have a sample per line and the individual sentences are seperated by a tab (\t). Some datasets (like AllNLI) has three sentences per line: An anchor, a positive, and a hard negative.

We measure for each dataset the performance on the STSb development dataset after 2k training steps with a distilroberta-base model and a batch size of 256. 

**Note**: We find that the STSb dataset is a suboptimal dataset to evaluate the quality of sentence embedding models. It consists mainly of rather simple sentences, it does not require any domain specific knowledge, and the included sentences are of rather high quality compared to noisy, user-written content. Please do not infer from the above numbers how the approaches will perform on your domain specific dataset.

## Training
See [training.py](training.py) for the training script.

The training script allows to load one or multiple files. We construct batches by sampling examples from the respective dataset. So far, examples are not mixed between the datasets, i.e., a batch consists only of examples from a single dataset.

As the dataset sizes are quite different in size, we perform a tempurate controlled sampling from the datasets: Smaller datasets are up-sampled, while larger datasets are down-sampled. This allows an effective training with very large and smaller datasets.

## Pre-Trained Models
Have a look at [pre-trained models](https://www.sbert.net/docs/pretrained_models.html) to view all models that were trained on these paraphrase datasets.

- **paraphrase-MiniLM-L12-v2** - Trained on the following datasets: AllNLI, sentence-compression, SimpleWiki, altlex, msmarco-triplets, quora_duplicates, coco_captions,flickr30k_captions, yahoo_answers_title_question, S2ORC_citation_pairs, stackexchange_duplicate_questions, wiki-atomic-edits
- **paraphrase-distilroberta-base-v2** - Trained on the following datasets: AllNLI, sentence-compression, SimpleWiki, altlex, msmarco-triplets, quora_duplicates, coco_captions,flickr30k_captions, yahoo_answers_title_question, S2ORC_citation_pairs, stackexchange_duplicate_questions, wiki-atomic-edits
- **paraphrase-distilroberta-base-v1** - Trained on the following datasets: AllNLI, sentence-compression, SimpleWiki, altlex, quora_duplicates, wiki-atomic-edits, wiki-split
- **paraphrase-xlm-r-multilingual-v1** - Multilingual version of paraphrase-distilroberta-base-v1, trained on parallel data for 50+ languages. (Teacher: paraphrase-distilroberta-base-v1, Student: xlm-r-base)


## Work in Progress

Training with this data is currently work-in-progress. Things that will be added in the next time:
- **More datasets**: Are you aware of more suitable training datasets? Let me know: [info@nils-reimers.de](mailto:info@nils-reimers.de)
- **Optimized batching**: Currently batches are only drawn from one dataset. Future work might include also batches that are sampled across datasets
- **Optimized loss function**: Currently the same parameters of MultipleNegativesRankingLoss is used for all datasets. Future work includes testing if the dataset benefit from individual loss functions.
- **Pre-trained models**: Once all datasets are collected, we will train and release respective models.