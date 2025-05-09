# Sparse Encoder Evaluation

This directory contains examples demonstrating how to evaluate Sparse Encoder models using various metrics and evaluator classes.

To run any of these evaluation scripts, simply execute the Python script. Each script will:
1. Load a pretrained sparse encoder model.
2. Prepare the evaluation dataset.
3. Configure the appropriate evaluator.
4. Run the evaluation.
5. Report the results.

```{eval-rst}
=============================================================================================  =========================================================================================================================================================================
Evaluator                                                                                      Evaluation Script
=============================================================================================  =========================================================================================================================================================================
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseInformationRetrievalEvaluator`  `sparse_retrieval_evaluator.py <https://github.com/UKPLab/sentence-transformers/blob/master/examples/sparse_encoder/evaluation/sparse_retrieval_evaluator.py>`_
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseNanoBEIREvaluator`              `sparse_nanobeir_evaluator.py <https://github.com/UKPLab/sentence-transformers/blob/master/examples/sparse_encoder/evaluation/sparse_nanobeir_evaluator.py>`_
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseEmbeddingSimilarityEvaluator`   `sparse_similarity_evaluator.py <https://github.com/UKPLab/sentence-transformers/blob/master/examples/sparse_encoder/evaluation/sparse_similarity_evaluator.py>`_
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseBinaryClassificationEvaluator`  `sparse_classification_evaluator.py <https://github.com/UKPLab/sentence-transformers/blob/master/examples/sparse_encoder/evaluation/sparse_classification_evaluator.py>`_
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseTripletEvaluator`               `sparse_triplet_evaluator.py <https://github.com/UKPLab/sentence-transformers/blob/master/examples/sparse_encoder/evaluation/sparse_triplet_evaluator.py>`_
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseRerankingEvaluator`             `sparse_reranking_evaluator.py <https://github.com/UKPLab/sentence-transformers/blob/master/examples/sparse_encoder/evaluation/sparse_reranking_evaluator.py>`_
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseTranslationEvaluator`           `sparse_translation_evaluator.py <https://github.com/UKPLab/sentence-transformers/blob/master/examples/sparse_encoder/evaluation/sparse_translation_evaluator.py>`_
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseMSEEvaluator`                   `sparse_mse_evaluator.py <https://github.com/UKPLab/sentence-transformers/blob/master/examples/sparse_encoder/evaluation/sparse_mse_evaluator.py>`_
=============================================================================================  =========================================================================================================================================================================
```

## Example with Retrieval Evaluation: 

This script demonstrates how to evaluate a sparse encoder on an information retrieval task ([`sparse_retrieval_evaluator.py`](sparse_retrieval_evaluator.py)):


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