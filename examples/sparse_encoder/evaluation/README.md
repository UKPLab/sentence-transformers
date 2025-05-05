# Sparse Encoder Evaluation

This directory contains examples demonstrating how to evaluate Sparse Encoder models using various metrics and evaluator classes.

## Usage

To run any of these evaluation scripts, simply execute the Python script. Each script will:
1. Load a pretrained sparse encoder model
2. Prepare the evaluation dataset
3. Configure the appropriate evaluator
4. Run the evaluation
5. Report the results

 
## 1. Retrieval Evaluation: 

This script demonstrates how to evaluate a sparse encoder on an information retrieval task ([`sparse_retrieval_evaluator.py`](sparse_retrieval_evaluator.py)):

<details>
<summary>Toggle Sparse Information Retrieval Evaluation Code</summary>

```python
import logging
import random

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseInformationRetrievalEvaluator

logging.basicConfig(format="%(message)s", level=logging.INFO)

# Load a model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# Load the NFcorpus IR dataset (https://huggingface.co/datasets/BeIR/nfcorpus, https://huggingface.co/datasets/BeIR/nfcorpus-qrels)
corpus = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
queries = load_dataset("BeIR/nfcorpus", "queries", split="queries")
relevant_docs_data = load_dataset("BeIR/nfcorpus-qrels", split="test")

# For this dataset, we want to concatenate the title and texts for the corpus
corpus = corpus.map(lambda x: {"text": x["title"] + " " + x["text"]}, remove_columns=["title"])

# Shrink the corpus size heavily to only the relevant documents + 1,000 random documents
required_corpus_ids = set(map(str, relevant_docs_data["corpus-id"]))
required_corpus_ids |= set(random.sample(corpus["_id"], k=1000))
corpus = corpus.filter(lambda x: x["_id"] in required_corpus_ids)

# Convert the datasets to dictionaries
corpus = dict(zip(corpus["_id"], corpus["text"]))  # Our corpus (cid => document)
queries = dict(zip(queries["_id"], queries["text"]))  # Our queries (qid => question)
relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
for qid, corpus_ids in zip(relevant_docs_data["query-id"], relevant_docs_data["corpus-id"]):
    qid = str(qid)
    corpus_ids = str(corpus_ids)
    if qid not in relevant_docs:
        relevant_docs[qid] = set()
    relevant_docs[qid].add(corpus_ids)

# Given queries, a corpus and a mapping with relevant documents, the SparseInformationRetrievalEvaluator computes different IR metrics.
ir_evaluator = SparseInformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="BeIR-nfcorpus-subset-test",
    show_progress_bar=True,
    batch_size=16,
)

# Run evaluation
results = ir_evaluator(model)
"""
Query info: num_rows: 323, num_cols: 30522, row_non_zero_mean: 42.891639709472656, row_sparsity_mean: 0.9985947012901306
Corpus info: num_rows: 3270, num_cols: 30522, row_non_zero_mean: 206.98899841308594, row_sparsity_mean: 0.9932184219360352
Score-Function: dot
Accuracy@1: 50.46%
Accuracy@3: 64.09%
Accuracy@5: 67.49%
Accuracy@10: 72.14%
Precision@1: 50.46%
Precision@3: 40.76%
Precision@5: 34.06%
Precision@10: 25.98%
Recall@1: 6.09%
Recall@3: 11.73%
Recall@5: 13.64%
Recall@10: 17.21%
MRR@10: 0.5796
NDCG@10: 0.3613
MAP@100: 0.1827
Primary metric value: 0.3613
"""
# Print the results
print(f"Primary metric: {ir_evaluator.primary_metric}")
# => Primary metric: BeIR-nfcorpus-subset-test_dot_ndcg@10
print(f"Primary metric value: {results[ir_evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.3613
```

</details>

## 2. Classification Evaluation:

This script shows how to evaluate a sparse encoder on classification tasks ([`sparse_classification_evaluator.py`](sparse_classification_evaluator.py)):

<details>
<summary>Toggle Sparse Classification Evaluation Code</summary>

```python
import logging

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseBinaryClassificationEvaluator

logging.basicConfig(format="%(message)s", level=logging.INFO)

# Initialize the SPLADE model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# Load a dataset with two text columns and a class label column (https://huggingface.co/datasets/sentence-transformers/quora-duplicates)
eval_dataset = load_dataset("sentence-transformers/quora-duplicates", "pair-class", split="train[-1000:]")

# Initialize the evaluator
binary_acc_evaluator = SparseBinaryClassificationEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    labels=eval_dataset["label"],
    name="quora_duplicates_dev",
    show_progress_bar=True,
    similarity_fn_names=["cosine", "dot", "euclidean", "manhattan"],
)
results = binary_acc_evaluator(model)
"""
Accuracy with Cosine-Similarity:             74.90      (Threshold: 0.8668)
F1 with Cosine-Similarity:                   67.37      (Threshold: 0.5959)
Precision with Cosine-Similarity:            54.15
Recall with Cosine-Similarity:               89.13
Average Precision with Cosine-Similarity:    67.81
Matthews Correlation with Cosine-Similarity: 49.89

Accuracy with Dot-Product:             76.50    (Threshold: 24.3460)
F1 with Dot-Product:                   66.93    (Threshold: 20.0762)
Precision with Dot-Product:            57.62
Recall with Dot-Product:               79.81
Average Precision with Dot-Product:    65.94
Matthews Correlation with Dot-Product: 48.82

Accuracy with Euclidean-Distance:             67.70     (Threshold: -10.0062)
F1 with Euclidean-Distance:                   48.60     (Threshold: -0.2346)
Precision with Euclidean-Distance:            32.13
Recall with Euclidean-Distance:               99.69
Average Precision with Euclidean-Distance:    20.52
Matthews Correlation with Euclidean-Distance: -4.59

Accuracy with Manhattan-Distance:             67.70     (Threshold: -103.1993)
F1 with Manhattan-Distance:                   48.60     (Threshold: -1.1565)
Precision with Manhattan-Distance:            32.13
Recall with Manhattan-Distance:               99.69
Average Precision with Manhattan-Distance:    21.05
Matthews Correlation with Manhattan-Distance: -4.59
"""
# Print the results
print(f"Primary metric: {binary_acc_evaluator.primary_metric}")
# => Primary metric: quora_duplicates_dev_max_ap
print(f"Primary metric value: {results[binary_acc_evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.6781
```

</details>

## 3. Similarity Evaluation:

This script shows how to evaluate a sparse encoder on semantic similarity tasks ([`sparse_similarity_evaluator.py`](sparse_similarity_evaluator.py)):

<details>
<summary>Toggle Sparse Similarity Evaluation Code</summary>

```python
import logging

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseEmbeddingSimilarityEvaluator

logging.basicConfig(format="%(message)s", level=logging.INFO)

# Load a model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# Load the STSB dataset (https://huggingface.co/datasets/sentence-transformers/stsb)
eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")

# Initialize the evaluator
dev_evaluator = SparseEmbeddingSimilarityEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    scores=eval_dataset["score"],
    name="sts_dev",
)
results = dev_evaluator(model)
"""
EmbeddingSimilarityEvaluator: Evaluating the model on the sts_dev dataset:
Dot-Similarity :	Pearson: 0.7513	Spearman: 0.8010
"""
# Print the results
print(f"Primary metric: {dev_evaluator.primary_metric}")
# => Primary metric: sts_dev_spearman_dot
print(f"Primary metric value: {results[dev_evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.8010
```

</details>

## 4. NanoBEIR Evaluation: 

These scripts demonstrate evaluation on a subset of NanoBEIR ([`sparse_nanobeir_evaluator.py`](sparse_nanobeir_evaluator.py)):

<details>
<summary>Toggle Sparse NanoBEIR Evaluation Code</summary>

```python
import logging

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

logging.basicConfig(format="%(message)s", level=logging.INFO)

# Load a model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

datasets = ["QuoraRetrieval", "MSMARCO"]

evaluator = SparseNanoBEIREvaluator(
    dataset_names=datasets,
    show_progress_bar=True,
    batch_size=32,
)

# Run evaluation
results = evaluator(model)
"""
Evaluating NanoQuoraRetrieval
Information Retrieval Evaluation of the model on the NanoQuoraRetrieval dataset:
Query info: num_rows: 50, num_cols: 30522, row_non_zero_mean: 62.97999954223633, row_sparsity_mean: 0.9979365468025208 1/1 [00:04<00:00,  4.12s/it]
Corpus info: num_rows: 5046, num_cols: 30522, row_non_zero_mean: 63.394371032714844, row_sparsity_mean: 0.9979230165481567
Score-Function: dot
Accuracy@1: 92.00%
Accuracy@3: 96.00%
Accuracy@5: 98.00%
Accuracy@10: 100.00%
Precision@1: 92.00%
Precision@3: 40.00%
Precision@5: 24.80%
Precision@10: 13.20%
Recall@1: 79.73%
Recall@3: 92.53%
Recall@5: 94.93%
Recall@10: 98.27%
MRR@10: 0.9439
NDCG@10: 0.9339
MAP@100: 0.9072

Evaluating NanoMSMARCO
Information Retrieval Evaluation of the model on the NanoMSMARCO dataset:
Query info: num_rows: 50, num_cols: 30522, row_non_zero_mean: 48.099998474121094, row_sparsity_mean: 0.99842399358749391/1 [00:19<00:00, 19.40s/it]
Corpus info: num_rows: 5043, num_cols: 30522, row_non_zero_mean: 125.38131713867188, row_sparsity_mean: 0.9958921670913696
Score-Function: dot
Accuracy@1: 48.00%
Accuracy@3: 74.00%
Accuracy@5: 76.00%
Accuracy@10: 88.00%
Precision@1: 48.00%
Precision@3: 24.67%
Precision@5: 15.20%
Precision@10: 8.80%
Recall@1: 48.00%
Recall@3: 74.00%
Recall@5: 76.00%
Recall@10: 88.00%
MRR@10: 0.6211
NDCG@10: 0.6838
MAP@100: 0.6277

Average Querie: num_rows: 50.0, num_cols: 30522.0, row_non_zero_mean: 55.53999900817871, row_sparsity_mean: 0.9981802701950073
Average Corpus: num_rows: 5044.5, num_cols: 30522.0, row_non_zero_mean: 94.38784408569336, row_sparsity_mean: 0.9969075918197632
Aggregated for Score Function: dot
Accuracy@1: 70.00%
Accuracy@3: 85.00%
Accuracy@5: 87.00%
Accuracy@10: 94.00%
Precision@1: 70.00%
Recall@1: 63.87%
Precision@3: 32.33%
Recall@3: 83.27%
Precision@5: 20.00%
Recall@5: 85.47%
Precision@10: 11.00%
Recall@10: 93.13%
MRR@10: 0.7825
NDCG@10: 0.8089
"""
# Print the results
print(f"Primary metric: {evaluator.primary_metric}")
# => Primary metric: NanoBEIR_mean_dot_ndcg@10
print(f"Primary metric value: {results[evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.8089
```

</details>

These scripts demonstrate evaluation on a NanoBEIR complete ([`sparse_nanobeir_advanced_evaluator.py`](sparse_nanobeir_advanced_evaluator.py)):

<details>
<summary>Toggle Sparse NanoBEIR Advanced Evaluation Code</summary>

```python
import logging

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

logging.basicConfig(format="%(message)s", level=logging.INFO)

# Load a model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")


evaluator = SparseNanoBEIREvaluator(
    dataset_names=None,  # None means evaluate on all datasets
    show_progress_bar=True,
    batch_size=16,
)

# Run evaluation
results = evaluator(model)
"""
Average Querie: num_rows: 49.92307692307692, num_cols: 30522.0, row_non_zero_mean: 74.91560451801007, row_sparsity_mean: 0.9975455219929035
Average Corpus: num_rows: 4334.7692307692305, num_cols: 30522.0, row_non_zero_mean: 174.81000049297626, row_sparsity_mean: 0.9942726905529315
Aggregated for Score Function: dot
Accuracy@1: 58.72%
Accuracy@3: 75.37%
Accuracy@5: 80.76%
Accuracy@10: 87.07%
Precision@1: 58.72%
Recall@1: 35.61%
Precision@3: 36.31%
Recall@3: 50.84%
Precision@5: 27.75%
Recall@5: 56.55%
Precision@10: 19.18%
Recall@10: 64.24%
MRR@10: 0.6821
NDCG@10: 0.6204
"""
# Print the results
print(f"Primary metric: {evaluator.primary_metric}")
# => Primary metric: NanoBEIR_mean_dot_ndcg@10
print(f"Primary metric value: {results[evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.6204
```

</details>

## 5. Reranking Evaluation: 

This script shows how to evaluate a sparse encoder for document reranking ([`sparse_reranking_evaluator.py`](sparse_reranking_evaluator.py)):

<details>
<summary>Toggle Sparse Reranking Evaluation Code</summary>

```python
import logging

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseRerankingEvaluator

logging.basicConfig(format="%(message)s", level=logging.INFO)

# Load a model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# Load a dataset with queries, positives, and negatives
eval_dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation").select(range(1000))

samples = [
    {
        "query": sample["query"],
        "positive": [
            text
            for is_selected, text in zip(sample["passages"]["is_selected"], sample["passages"]["passage_text"])
            if is_selected
        ],
        "negative": [
            text
            for is_selected, text in zip(sample["passages"]["is_selected"], sample["passages"]["passage_text"])
            if not is_selected
        ],
    }
    for sample in eval_dataset
]


# Now evaluate using only the documents from the 1000 samples
reranking_evaluator = SparseRerankingEvaluator(
    samples=samples,
    name="ms-marco-dev-small",
    show_progress_bar=True,
    batch_size=32,
)

results = reranking_evaluator(model)
"""
RerankingEvaluator: Evaluating the model on the ms-marco-dev-small dataset:
Queries: 967 	 Positives: Min 1.0, Mean 1.1, Max 3.0 	 Negatives: Min 1.0, Mean 7.1, Max 9.0
MAP: 53.46
MRR@10: 54.18
NDCG@10: 65.10
"""
# Print the results
print(f"Primary metric: {reranking_evaluator.primary_metric}")
# => Primary metric: ms-marco-dev-small_ndcg@10
print(f"Primary metric value: {results[reranking_evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.6510
```

</details>


## 6. MSE Evaluation: 

This script demonstrates mean squared error evaluation for sparse encoders ([`sparse_mse_evaluator.py`](sparse_mse_evaluator.py)):

<details>
<summary>Toggle Sparse MSE Evaluation Code</summary>

```python
import logging

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseMSEEvaluator

logging.basicConfig(format="%(message)s", level=logging.INFO)

# Load a model
student_model = SparseEncoder("prithivida/Splade_PP_en_v1")
teacher_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# Load any dataset with some texts
dataset = load_dataset("sentence-transformers/stsb", split="validation")
sentences = dataset["sentence1"] + dataset["sentence2"]

# Given queries, a corpus and a mapping with relevant documents, the SparseMSEEvaluator computes different MSE metrics.
mse_evaluator = SparseMSEEvaluator(
    source_sentences=sentences,
    target_sentences=sentences,
    teacher_model=teacher_model,
    name="stsb-dev",
)
results = mse_evaluator(student_model)
"""
MSE evaluation (lower = better) on the stsb-dev dataset:
MSE (*100):	0.035540
"""
# Print the results
print(f"Primary metric: {mse_evaluator.primary_metric}")
# => Primary metric: stsb-dev_negative_mse
print(f"Primary metric value: {results[mse_evaluator.primary_metric]:.4f}")
# => Primary metric value: -0.0355
```

</details>

## 7. Triplet Evaluation: 

This script shows how to evaluate a sparse encoder on triplet-based tasks([`sparse_triplet_evaluator.py`](sparse_triplet_evaluator.py)
):

<details>
<summary>Toggle Sparse Triplet Evaluation Code</summary>

```python
import logging

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseTripletEvaluator

logging.basicConfig(format="%(message)s", level=logging.INFO)

# Load a model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# Load triplets from the AllNLI dataset
# The dataset contains triplets of (anchor, positive, negative) sentences
dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev[:1000]")

# Initialize the SparseTripletEvaluator
evaluator = SparseTripletEvaluator(
    anchors=dataset[:1000]["anchor"],
    positives=dataset[:1000]["positive"],
    negatives=dataset[:1000]["negative"],
    name="all_nli_dev",
    batch_size=32,
    show_progress_bar=True,
)

# Run the evaluation
results = evaluator(model)
"""
TripletEvaluator: Evaluating the model on the all_nli_dev dataset:
Accuracy Dot Similarity:	85.10%
"""
# Print the results
print(f"Primary metric: {evaluator.primary_metric}")
# => Primary metric: all_nli_dev_dot_accuracy
print(f"Primary metric value: {results[evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.8510
```

</details>

## 8. Translation Evaluation: 

This script demonstrates evaluation on cross-lingual tasks ([`sparse_translation_evaluator.py`](sparse_translation_evaluator.py)):

<details>
<summary>Toggle Sparse Translation Evaluation Code</summary>

```python
import logging

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseTranslationEvaluator

logging.basicConfig(format="%(message)s", level=logging.INFO)

# Load a model, not mutilingual but hope to see some on the hub soon
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# Load a parallel sentences dataset
dataset = load_dataset("sentence-transformers/parallel-sentences-news-commentary", "en-nl", split="train[:1000]")

# Initialize the TranslationEvaluator using the same texts from two languages
translation_evaluator = SparseTranslationEvaluator(
    source_sentences=dataset["english"],
    target_sentences=dataset["non_english"],
    name="news-commentary-en-nl",
)
results = translation_evaluator(model)
"""
Evaluating translation matching Accuracy of the model on the news-commentary-en-nl dataset:
Accuracy src2trg: 41.40
Accuracy trg2src: 47.70
"""
# Print the results
print(f"Primary metric: {translation_evaluator.primary_metric}")
# => Primary metric: news-commentary-en-nl_mean_accuracy
print(f"Primary metric value: {results[translation_evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.4455
```

</details>
