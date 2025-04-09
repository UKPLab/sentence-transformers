from __future__ import annotations

from sentence_transformers.evaluation.NanoBEIREvaluator import (
    DatasetNameType,
    NanoBEIREvaluator,
    dataset_name_to_id,
)
from sentence_transformers.sparse_encoder.evaluation.SparseInformationRetrievalEvaluator import (
    SparseInformationRetrievalEvaluator,
)
from sentence_transformers.util import is_datasets_available


class SparseNanoBEIREvaluator(NanoBEIREvaluator):
    """
    This class evaluates the performance of a SparseEncoder Model on the NanoBEIR collection of Information Retrieval datasets.

    The collection is a set of datasets based on the BEIR collection, but with a significantly smaller size, so it can
    be used for quickly evaluating the retrieval performance of a model before commiting to a full evaluation.
    The datasets are available on Hugging Face in the `NanoBEIR collection <https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6>`_.
    This evaluator will return the same metrics as the SparseInformationRetrievalEvaluator (i.e., MRR, nDCG, Recall@k), for each dataset and on average.

    Args:
        dataset_names (List[str]): The names of the datasets to evaluate on. Defaults to all datasets.
        mrr_at_k (List[int]): A list of integers representing the values of k for MRR calculation. Defaults to [10].
        ndcg_at_k (List[int]): A list of integers representing the values of k for NDCG calculation. Defaults to [10].
        accuracy_at_k (List[int]): A list of integers representing the values of k for accuracy calculation. Defaults to [1, 3, 5, 10].
        precision_recall_at_k (List[int]): A list of integers representing the values of k for precision and recall calculation. Defaults to [1, 3, 5, 10].
        map_at_k (List[int]): A list of integers representing the values of k for MAP calculation. Defaults to [100].
        show_progress_bar (bool): Whether to show a progress bar during evaluation. Defaults to False.
        batch_size (int): The batch size for evaluation. Defaults to 32.
        write_csv (bool): Whether to write the evaluation results to a CSV file. Defaults to True.
        truncate_dim (int, optional): The dimension to truncate the embeddings to. Defaults to None.
        score_functions (Dict[str, Callable[[Tensor, Tensor], Tensor]]): A dictionary mapping score function names to score functions. Defaults to {SimilarityFunction.COSINE.value: cos_sim, SimilarityFunction.DOT_PRODUCT.value: dot_score}.
        main_score_function (Union[str, SimilarityFunction], optional): The main score function to use for evaluation. Defaults to None.
        aggregate_fn (Callable[[list[float]], float]): The function to aggregate the scores. Defaults to np.mean.
        aggregate_key (str): The key to use for the aggregated score. Defaults to "mean".
        query_prompts (str | dict[str, str], optional): The prompts to add to the queries. If a string, will add the same prompt to all queries. If a dict, expects that all datasets in dataset_names are keys.
        corpus_prompts (str | dict[str, str], optional): The prompts to add to the corpus. If a string, will add the same prompt to all corpus. If a dict, expects that all datasets in dataset_names are keys.

    Example:
        ::

            from sentence_transformers.models import Pooling, Transformer
            from sentence_transformers.sparse_encoder import SparseEncoder
            from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator
            from sentence_transformers.sparse_encoder.models import CSRSparsity

            # Initialize model components
            model_name = "sentence-transformers/all-mpnet-base-v2"
            transformer = Transformer(model_name)
            pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
            csr_sparsity = CSRSparsity(
                input_dim=transformer.get_word_embedding_dimension(),
                hidden_dim=4 * transformer.get_word_embedding_dimension(),
                k=32,  # Number of top values to keep
                k_aux=512,  # Number of top values for auxiliary loss
            )
            # Create the SparseEncoder model
            model = SparseEncoder(modules=[transformer, pooling, csr_sparsity])

            # Load & run the evaluator
            dataset_names = ["msmarco", "nfcorpus", "nq"]
            evaluator = SparseNanoBEIREvaluator(dataset_names)
            results = evaluator(model)
            '''
            NanoBEIR Evaluation of the model on ['msmarco', 'nfcorpus', 'nq'] dataset:
            Evaluating NanoMSMARCO
            SparseInformationRetrievalEvaluator: Evaluating the model on the NanoMSMARCO dataset:
            Queries: 50
            Corpus: 5043

            Score-Function: cosine
            Accuracy@1: 40.00%
            Accuracy@3: 74.00%
            Accuracy@5: 78.00%
            Accuracy@10: 88.00%
            Precision@1: 40.00%
            Precision@3: 24.67%
            Precision@5: 15.60%
            Precision@10: 8.80%
            Recall@1: 40.00%
            Recall@3: 74.00%
            Recall@5: 78.00%
            Recall@10: 88.00%
            MRR@10: 0.5849
            NDCG@10: 0.6572
            MAP@100: 0.5892
            '''
            print(evaluator.primary_metric)
            # => "NanoBEIR_mean_cosine_ndcg@10"
            print(results[evaluator.primary_metric])
            # => 0.6572
    """

    def _load_dataset(
        self, dataset_name: DatasetNameType, **ir_evaluator_kwargs
    ) -> SparseInformationRetrievalEvaluator:
        if not is_datasets_available():
            raise ValueError(
                "datasets is not available. Please install it to use the SparseNanoBEIREvaluator via `pip install datasets`."
            )
        from datasets import load_dataset

        dataset_path = dataset_name_to_id[dataset_name.lower()]
        corpus = load_dataset(dataset_path, "corpus", split="train")
        queries = load_dataset(dataset_path, "queries", split="train")
        qrels = load_dataset(dataset_path, "qrels", split="train")
        corpus_dict = {sample["_id"]: sample["text"] for sample in corpus if len(sample["text"]) > 0}
        queries_dict = {sample["_id"]: sample["text"] for sample in queries if len(sample["text"]) > 0}
        qrels_dict = {}
        for sample in qrels:
            if sample["query-id"] not in qrels_dict:
                qrels_dict[sample["query-id"]] = set()
            qrels_dict[sample["query-id"]].add(sample["corpus-id"])

        if self.query_prompts is not None:
            ir_evaluator_kwargs["query_prompt"] = self.query_prompts.get(dataset_name, None)
        if self.corpus_prompts is not None:
            ir_evaluator_kwargs["corpus_prompt"] = self.corpus_prompts.get(dataset_name, None)
        human_readable_name = self._get_human_readable_name(dataset_name)
        return SparseInformationRetrievalEvaluator(
            queries=queries_dict,
            corpus=corpus_dict,
            relevant_docs=qrels_dict,
            name=human_readable_name,
            **ir_evaluator_kwargs,
        )
