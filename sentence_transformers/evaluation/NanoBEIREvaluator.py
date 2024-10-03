from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Callable

import numpy as np
from torch import Tensor
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation.InformationRetrievalEvaluator import InformationRetrievalEvaluator
from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.util import cos_sim, dot_score, is_datasets_available

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


dataset_paths = {
    "climatefever": "zeta-alpha-ai/NanoClimateFEVER",
    "dbpedia": "zeta-alpha-ai/NanoDBPedia",
    "fever": "zeta-alpha-ai/NanoFEVER",
    "fiqa2018": "zeta-alpha-ai/NanoFiQA2018",
    "hotpotqa": "zeta-alpha-ai/NanoHotpotQA",
    "msmarco": "zeta-alpha-ai/NanoMSMARCO",
    "nfcorpus": "zeta-alpha-ai/NanoNFCorpus",
    "nq": "zeta-alpha-ai/NanoNQ",
    "quoraretrieval": "zeta-alpha-ai/NanoQuoraRetrieval",
    "scidocs": "zeta-alpha-ai/NanoSCIDOCS",
    "arguana": "zeta-alpha-ai/NanoArguAna",
    "scifact": "zeta-alpha-ai/NanoSciFact",
    "touche2020": "zeta-alpha-ai/NanoTouche2020",
}

clean_names = {
    "zeta-alpha-ai/NanoClimateFEVER": "ClimateFEVER",
    "zeta-alpha-ai/NanoDBPedia": "DBPedia",
    "zeta-alpha-ai/NanoFEVER": "FEVER",
    "zeta-alpha-ai/NanoFiQA2018": "FiQA2018",
    "zeta-alpha-ai/NanoHotpotQA": "HotpotQA",
    "zeta-alpha-ai/NanoMSMARCO": "MSMARCO",
    "zeta-alpha-ai/NanoNFCorpus": "NFCorpus",
    "zeta-alpha-ai/NanoNQ": "NQ",
    "zeta-alpha-ai/NanoQuoraRetrieval": "QuoraRetrieval",
    "zeta-alpha-ai/NanoSCIDOCS": "SCIDOCS",
    "zeta-alpha-ai/NanoArguAna": "ArguAna",
    "zeta-alpha-ai/NanoSciFact": "SciFact",
    "zeta-alpha-ai/NanoTouche2020": "Touche2020",
}


class NanoBeIREvaluator(SentenceEvaluator):
    """
    This class evaluates the performance of a SentenceTransformer Model on the NanoBEIR collection of datasets.

    The collection is a set of datasets based on the BEIR collection, but with a significantly smaller size, so it can be used for quickly evaluating the retrieval performance of a model before commiting to a full evaluation.
    The datasets are available on HuggingFace at https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6
    The Evaluator will return the same metrics as the InformationRetrievalEvaluator (i.e., MRR, nDCG, Recall@k), for each dataset and on average.


    Example:
        ::

            from sentence_transformers import SentenceTransformer
            from sentence_transformers.evaluation import NanoBEIREvaluator

            # Load a model
            model = SentenceTransformer('all-mpnet-base-v2')

            datasets = ["QuoraRetrieval", "MSMARCO"]
            query_prompts = {
                "QuoraRetrieval": "Instruct: Given a question, retrieve questions that are semantically equivalent to the given question\nQuery: ",
                "MSMARCO": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
            }

            evaluator = NanoBEIREvaluator(
                dataset_names=datasets,
                name="NanoBEIR",
                query_prompts=query_prompts,
            )

            results = evaluator(model)
            '''
            NanoBEeIR Evaluation of the model on ['QuoraRetrieval', 'MSMARCO'] dataset:
            Evaluating NanoBeIRNanoQuoraRetrieval
            Evaluating NanoBeIRNanoMSMARCO

            Average Queries: 50.0
            Average Corpus: 5044.5

            Aggregated for Score Function: cosine
            Accuracy@1: 39.00%
            Accuracy@3: 57.00%
            Accuracy@5: 66.00%
            Accuracy@10: 77.00%
            Precision@1: 39.00%
            Recall@1: 34.03%
            Precision@3: 20.67%
            Recall@3: 54.07%
            Precision@5: 15.00%
            Recall@5: 64.27%
            Precision@10: 8.90%
            Recall@10: 75.97%
            MRR@10: 0.5004
            NDCG@10: 0.5513
            Aggregated for Score Function: dot
            Accuracy@1: 39.00%
            Accuracy@3: 57.00%
            Accuracy@5: 66.00%
            Accuracy@10: 77.00%
            Precision@1: 39.00%
            Recall@1: 34.03%
            Precision@3: 20.67%
            Recall@3: 54.07%
            Precision@5: 15.00%
            Recall@5: 64.27%
            Precision@10: 8.90%
            Recall@10: 75.97%
            MRR@10: 0.5004
            NDCG@10: 0.5513
            '''
            logger.info(evaluator.primary_metric)
            # => "cosine_ndcg@10"
            logger.info(results["mean"][evaluator.primary_metric])
            # => 0.5512516989358924
    """

    def __init__(
        self,
        dataset_names: list[str] | None = None,
        mrr_at_k: list[int] = [10],
        ndcg_at_k: list[int] = [10],
        accuracy_at_k: list[int] = [1, 3, 5, 10],
        precision_recall_at_k: list[int] = [1, 3, 5, 10],
        map_at_k: list[int] = [100],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = "",
        write_csv: bool = True,
        truncate_dim: int | None = None,
        score_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
            SimilarityFunction.COSINE.value: cos_sim,
            SimilarityFunction.DOT_PRODUCT.value: dot_score,
        },  # Score function, higher=more similar
        main_score_function: str | SimilarityFunction | None = None,
        aggregate_fn: Callable[[list[float]], float] = np.mean,
        aggregate_key: str = "mean",
        query_prompts: str | dict[str, str] | None = None,
        corpus_prompts: str | dict[str, str] | None = None,
    ):
        """
        Initializes the NanoBEIREvaluator.

        Args:
            dataset_names (List[str]): The names of the datasets to evaluate on.
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
        """
        super().__init__()
        if dataset_names is None:
            dataset_names = list(dataset_paths.keys())
        self.dataset_names = dataset_names
        self.aggregate_fn = aggregate_fn
        self.aggregate_key = aggregate_key
        self.write_csv = write_csv
        self.query_prompts = query_prompts
        self.corpus_prompts = corpus_prompts
        self.show_progress_bar = show_progress_bar
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys()))
        self.main_score_function = main_score_function
        self.truncate_dim = truncate_dim

        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        self._validate_dataset_names()
        self._validate_prompts()

        ir_evaluator_kwargs = {
            "mrr_at_k": mrr_at_k,
            "ndcg_at_k": ndcg_at_k,
            "accuracy_at_k": accuracy_at_k,
            "precision_recall_at_k": precision_recall_at_k,
            "map_at_k": map_at_k,
            "show_progress_bar": show_progress_bar,
            "batch_size": batch_size,
            "write_csv": write_csv,
            "truncate_dim": truncate_dim,
            "score_functions": score_functions,
            "main_score_function": main_score_function,
        }

        self.name = name

        self.evaluators = [self._load_dataset(name, **ir_evaluator_kwargs) for name in self.dataset_names]

        self.csv_file: str = f"NanoBEIR_evaluation_{aggregate_key}{self.name}_results.csv"
        self.csv_headers = ["epoch", "steps"]

        for score_name in self.score_function_names:
            for k in accuracy_at_k:
                self.csv_headers.append(f"{score_name}-Accuracy@{k}")

            for k in precision_recall_at_k:
                self.csv_headers.append(f"{score_name}-Precision@{k}")
                self.csv_headers.append(f"{score_name}-Recall@{k}")

            for k in mrr_at_k:
                self.csv_headers.append(f"{score_name}-MRR@{k}")

            for k in ndcg_at_k:
                self.csv_headers.append(f"{score_name}-NDCG@{k}")

            for k in map_at_k:
                self.csv_headers.append(f"{score_name}-MAP@{k}")

    def __call__(
        self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs
    ) -> dict[str, float]:
        per_metric_results = {}
        per_dataset_results = {}
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"
        logger.info(f"NanoBEeIR Evaluation of the model on {self.dataset_names} dataset{out_txt}:")
        for evaluator in tqdm(self.evaluators, desc="Evaluating datasets", disable=not self.show_progress_bar):
            logger.info(f"Evaluating {evaluator.name}")
            evaluation = evaluator(model, output_path, epoch, steps)
            for k in evaluation:
                dataset, metric = k.split("_", maxsplit=1)
                if metric not in per_metric_results:
                    per_metric_results[metric] = []
                if dataset not in per_dataset_results:
                    per_dataset_results[dataset] = {}
                per_dataset_results[dataset][metric] = evaluation[k]
                per_metric_results[metric].append(evaluation[k])
        per_dataset_results[self.aggregate_key] = {}
        for metric in per_metric_results:
            per_dataset_results[self.aggregate_key][metric] = self.aggregate_fn(per_metric_results[metric])

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.accuracy_at_k:
                    output_data.append(per_dataset_results[name]["accuracy@k"][k])

                for k in self.precision_recall_at_k:
                    output_data.append(per_dataset_results[name]["precision@k"][k])
                    output_data.append(per_dataset_results[name]["recall@k"][k])

                for k in self.mrr_at_k:
                    output_data.append(per_dataset_results[name]["mrr@k"][k])

                for k in self.ndcg_at_k:
                    output_data.append(per_dataset_results[name]["ndcg@k"][k])

                for k in self.map_at_k:
                    output_data.append(per_dataset_results[name]["map@k"][k])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        agg_results = per_dataset_results[self.aggregate_key]
        if not self.primary_metric:
            if self.main_score_function is None:
                score_function = max(
                    [(name, agg_results[f"{name}_ndcg@{max(self.ndcg_at_k)}"]) for name in self.score_function_names],
                    key=lambda x: x[1],
                )[0]
                self.primary_metric = f"{score_function}_ndcg@{max(self.ndcg_at_k)}"
            else:
                self.primary_metric = f"{self.main_score_function.value}_ndcg@{max(self.ndcg_at_k)}"

        self.store_metrics_in_model_card_data(model, agg_results)

        avg_queries = np.mean([len(evaluator.queries) for evaluator in self.evaluators])
        avg_corpus = np.mean([len(evaluator.corpus) for evaluator in self.evaluators])
        logger.info(f"\nAverage Queries: {avg_queries}")
        logger.info(f"Average Corpus: {avg_corpus}\n")

        scores = per_dataset_results[self.aggregate_key]
        for name in self.score_function_names:
            logger.info(f"Aggregated for Score Function: {name}")
            for k in self.accuracy_at_k:
                logger.info("Accuracy@{}: {:.2f}%".format(k, scores[f"{name}_accuracy@{k}"] * 100))

            for k in self.precision_recall_at_k:
                logger.info("Precision@{}: {:.2f}%".format(k, scores[f"{name}_precision@{k}"] * 100))
                logger.info("Recall@{}: {:.2f}%".format(k, scores[f"{name}_recall@{k}"] * 100))

            for k in self.mrr_at_k:
                logger.info("MRR@{}: {:.4f}".format(k, scores[f"{name}_mrr@{k}"]))

            for k in self.ndcg_at_k:
                logger.info("NDCG@{}: {:.4f}".format(k, scores[f"{name}_ndcg@{k}"]))
        return per_dataset_results

    def __get_clean_name(self, dataset_name: str) -> str:
        return f"Nano{clean_names[dataset_paths[dataset_name.lower()]]}"

    def _load_dataset(self, dataset_name: str, **ir_evaluator_kwargs) -> InformationRetrievalEvaluator:
        if not is_datasets_available():
            raise ValueError("datasets is not available. Please install it to use the NanoBEIREvaluator.")
        from datasets import load_dataset

        dataset_path = dataset_paths[dataset_name.lower()]
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
        clean_name = self.__get_clean_name(dataset_name)
        return InformationRetrievalEvaluator(
            queries=queries_dict,
            corpus=corpus_dict,
            relevant_docs=qrels_dict,
            name=f"{self.name}{clean_name}",
            **ir_evaluator_kwargs,
        )

    def _validate_dataset_names(self):
        missing_datasets = []
        for dataset_name in self.dataset_names:
            if dataset_name.lower() not in dataset_paths:
                missing_datasets.append(dataset_name)
        if missing_datasets:
            raise ValueError(
                f"Dataset(s) {missing_datasets} not found in NanoBEIR collection."
                f"Valid dataset names are: {dataset_paths.keys()}"
            )

    def _validate_prompts(self):
        missing_query_prompts = []
        missing_corpus_prompts = []
        for dataset_name in self.dataset_names:
            if self.query_prompts is not None and dataset_name not in self.query_prompts:
                missing_query_prompts.append(dataset_name)
            if self.corpus_prompts is not None and dataset_name not in self.corpus_prompts:
                missing_corpus_prompts.append(dataset_name)
        warning_msg = ""
        if missing_query_prompts:
            warning_msg += f"The following datasets are missing query prompts: {missing_query_prompts}\n"
        if missing_corpus_prompts:
            warning_msg += f"The following datasets are missing corpus prompts: {missing_corpus_prompts}\n"
        if warning_msg:
            raise ValueError(warning_msg)
