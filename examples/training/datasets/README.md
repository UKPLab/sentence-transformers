# Training Datasets

Most dataset configurations will take one of four forms:

- **Case 1**: The example is a pair of sentences and a label indicating how similar they are. The label can be either an integer or a float. This case applies to datasets originally prepared for Natural Language Inference (NLI), since they contain pairs of sentences with a label indicating whether they infer each other or not.
   **Case Example:** [SNLI](https://huggingface.co/datasets/snli).
- **Case 2**: The example is a pair of positive (similar) sentences **without** a label. For example, pairs of paraphrases, pairs of full texts and their summaries, pairs of duplicate questions, pairs of (`query`, `response`), or pairs of (`source_language`, `target_language`). Natural Language Inference datasets can also be formatted this way by pairing entailing sentences.
   **Case Examples:** [Sentence Compression](https://huggingface.co/datasets/embedding-data/sentence-compression), [COCO Captions](https://huggingface.co/datasets/embedding-data/coco_captions_quintets), [Flickr30k captions](https://huggingface.co/datasets/embedding-data/flickr30k_captions_quintets).
- **Case 3**: The example is a sentence with an integer label indicating the class to which it belongs. This data format is easily converted by loss functions into three sentences (triplets) where the first is an "anchor", the second a "positive" of the same class as the anchor, and the third a "negative" of a different class.
   **Case Examples:** [TREC](https://huggingface.co/datasets/trec), [Yahoo Answers Topics](https://huggingface.co/datasets/yahoo_answers_topics).
- **Case 4**: The example is a triplet (anchor, positive, negative) without classes or labels for the sentences.
   **Case Example:** [Quora Triplets](https://huggingface.co/datasets/embedding-data/QQP_triplets)

Note that Sentence Transformers models can be trained with human labeling (cases 1 and 3) or with labels automatically deduced from text formatting (cases 2 and 4).

You can get almost ready-to-train datasets from various sources. One of them is the Hugging Face Hub.

## Datasets on the Hugging Face Hub

The [Datasets library](https://huggingface.co/docs/datasets/index) (`pip install datasets`) allows you to load datasets from the Hugging Face Hub with the `load_dataset` function:

```python
from datasets import load_dataset

# Indicate the repo id from the Hub
dataset_id = "embedding-data/QQP_triplets"

dataset = load_dataset(dataset_id)
```

For more information on how to manipulate your dataset see [» Datasets Documentation](https://huggingface.co/docs/datasets/access).

These are popular datasets used to train and fine-tune SentenceTransformers models.

|   | Dataset                                                                                                   |
| - | --------------------------------------------------------------------------------------------------------- |
|   | [altlex pairs](https://huggingface.co/datasets/embedding-data/altlex)                                     |
|   | [sentence compression pairs](https://huggingface.co/datasets/embedding-data/sentence-compression)         |
|   | [QQP triplets](https://huggingface.co/datasets/embedding-data/QQP_triplets)                               |
|   | [PAQ pairs](https://huggingface.co/datasets/embedding-data/PAQ_pairs)                                     |
|   | [SPECTER triplets](https://huggingface.co/datasets/embedding-data/SPECTER)                                |
|   | [Amazon QA pairs](https://huggingface.co/datasets/embedding-data/Amazon-QA)                               |
|   | [Simple Wiki pairs](https://huggingface.co/datasets/embedding-data/simple-wiki)                           |
|   | [Wiki Answers equivalent sentences](https://huggingface.co/datasets/embedding-data/WikiAnswers)           |
|   | [COCO Captions quintets](https://huggingface.co/datasets/embedding-data/coco_captions_quintets)           |
|   | [Flickr30k Captions quintets](https://huggingface.co/datasets/embedding-data/flickr30k_captions_quintets) |
|   | [MS Marco](https://huggingface.co/datasets/ms_marco)                                                      |
|   | [GOOAQ](https://huggingface.co/datasets/gooaq)                                                            |
|   | [MS Marco](https://huggingface.co/datasets/ms_marco)                                                      |
|   | [Yahoo Answers topics](https://huggingface.co/datasets/yahoo_answers_topics)                              |
|   | [Search QA](https://huggingface.co/datasets/search_qa)                                                    |
|   | [Stack Exchange](https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_xml )             |
|   | [ELI5](https://huggingface.co/datasets/eli5)                                                              |
|   | [MultiNLI](https://huggingface.co/datasets/multi_nli)                                                     |
|   | [SNLI](https://huggingface.co/datasets/snli)                                                              |
|   | [S2ORC](https://huggingface.co/datasets/s2orc)                                                            |
|   | [Trivia QA](https://huggingface.co/datasets/trivia_qa)                                                    |
|   | [Code Search Net](https://huggingface.co/datasets/code_search_net)                                        |
|   | [Natural Questions](https://huggingface.co/datasets/natural_questions)                                    |
