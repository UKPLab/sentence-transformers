from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
from torch import Tensor
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation.InformationRetrievalEvaluator import InformationRetrievalEvaluator
from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.util import is_datasets_available

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)

DatasetNameType = Literal[
    "climatefever",
    "dbpedia",
    "fever",
    "fiqa2018",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quoraretrieval",
    "scidocs",
    "arguana",
    "scifact",
    "touche2020",
]


dataset_name_to_id = {
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

dataset_name_to_human_readable = {
    "climatefever": "ClimateFEVER",
    "dbpedia": "DBPedia",
    "fever": "FEVER",
    "fiqa2018": "FiQA2018",
    "hotpotqa": "HotpotQA",
    "msmarco": "MSMARCO",
    "nfcorpus": "NFCorpus",
    "nq": "NQ",
    "quoraretrieval": "QuoraRetrieval",
    "scidocs": "SCIDOCS",
    "arguana": "ArguAna",
    "scifact": "SciFact",
    "touche2020": "Touche2020",
}


class NanoBEIREvaluator(SentenceEvaluator):
    """
    This class evaluates the performance of a SentenceTransformer Model on the NanoBEIR collection of datasets.

    The collection is a set of datasets based on the BEIR collection, but with a significantly smaller size, so it can be used for quickly evaluating the retrieval performance of a model before commiting to a full evaluation.
    The datasets are available on HuggingFace at https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6
    The Evaluator will return the same metrics as the InformationRetrievalEvaluator (i.e., MRR, nDCG, Recall@k), for each dataset and on average.


    Example:
        ::

            from sentence_transformers import SentenceTransformer
            from sentence_transformers.evaluation import NanoBEIREvaluator

            model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

            datasets = ["QuoraRetrieval", "MSMARCO"]
            query_prompts = {
                "QuoraRetrieval": "Instruct: Given a question, retrieve questions that are semantically equivalent to the given question\\nQuery: ",
                "MSMARCO": "Instruct: Given a web search query, retrieve relevant passages that answer the query\\nQuery: "
            }

            evaluator = NanoBEIREvaluator(
                dataset_names=datasets,
                query_prompts=query_prompts,
            )

            results = evaluator(model)
            '''
            NanoBEIR Evaluation of the model on ['QuoraRetrieval', 'MSMARCO'] dataset:
            Evaluating NanoQuoraRetrieval
            Information Retrieval Evaluation of the model on the NanoQuoraRetrieval dataset:
            Queries: 50
            Corpus: 5046

            Score-Function: cosine
            Accuracy@1: 92.00%
            Accuracy@3: 98.00%
            Accuracy@5: 100.00%
            Accuracy@10: 100.00%
            Precision@1: 92.00%
            Precision@3: 40.67%
            Precision@5: 26.00%
            Precision@10: 14.00%
            Recall@1: 81.73%
            Recall@3: 94.20%
            Recall@5: 97.93%
            Recall@10: 100.00%
            MRR@10: 0.9540
            NDCG@10: 0.9597
            MAP@100: 0.9395

            Evaluating NanoMSMARCO
            Information Retrieval Evaluation of the model on the NanoMSMARCO dataset:
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
            Average Queries: 50.0
            Average Corpus: 5044.5

            Aggregated for Score Function: cosine
            Accuracy@1: 66.00%
            Accuracy@3: 86.00%
            Accuracy@5: 89.00%
            Accuracy@10: 94.00%
            Precision@1: 66.00%
            Recall@1: 60.87%
            Precision@3: 32.67%
            Recall@3: 84.10%
            Precision@5: 20.80%
            Recall@5: 87.97%
            Precision@10: 11.40%
            Recall@10: 94.00%
            MRR@10: 0.7694
            NDCG@10: 0.8085
            '''
            print(evaluator.primary_metric)
            # => "NanoBEIR_mean_cosine_ndcg@10"
            print(results[evaluator.primary_metric])
            # => 0.8084508771660436
    """

    def __init__(
        self,
        dataset_names: list[DatasetNameType] | None = None,
        mrr_at_k: list[int] = [10],
        ndcg_at_k: list[int] = [10],
        accuracy_at_k: list[int] = [1, 3, 5, 10],
        precision_recall_at_k: list[int] = [1, 3, 5, 10],
        map_at_k: list[int] = [100],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        write_csv: bool = True,
        truncate_dim: int | None = None,
        score_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] = None,
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
            dataset_names = list(dataset_name_to_id.keys())
        self.dataset_names = dataset_names
        self.aggregate_fn = aggregate_fn
        self.aggregate_key = aggregate_key
        self.write_csv = write_csv
        self.query_prompts = query_prompts
        self.corpus_prompts = corpus_prompts
        self.show_progress_bar = show_progress_bar
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys())) if score_functions else []
        self.main_score_function = main_score_function
        self.truncate_dim = truncate_dim
        self.name = f"NanoBEIR_{aggregate_key}"
        if self.truncate_dim:
            self.name += f"_{self.truncate_dim}"

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

        self.evaluators = [self._load_dataset(name, **ir_evaluator_kwargs) for name in self.dataset_names]

        self.csv_file: str = f"NanoBEIR_evaluation_{aggregate_key}_results.csv"
        self.csv_headers = ["epoch", "steps"]

        self._append_csv_headers(self.score_function_names)

    def _append_csv_headers(self, score_function_names):
        for score_name in score_function_names:
            for k in self.accuracy_at_k:
                self.csv_headers.append(f"{score_name}-Accuracy@{k}")

            for k in self.precision_recall_at_k:
                self.csv_headers.append(f"{score_name}-Precision@{k}")
                self.csv_headers.append(f"{score_name}-Recall@{k}")

            for k in self.mrr_at_k:
                self.csv_headers.append(f"{score_name}-MRR@{k}")

            for k in self.ndcg_at_k:
                self.csv_headers.append(f"{score_name}-NDCG@{k}")

            for k in self.map_at_k:
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
        logger.info(f"NanoBEIR Evaluation of the model on {self.dataset_names} dataset{out_txt}:")

        if self.score_functions is None:
            self.score_functions = {model.similarity_fn_name: model.similarity}
            self.score_function_names = [model.similarity_fn_name]
            self._append_csv_headers(self.score_function_names)

        for evaluator in tqdm(self.evaluators, desc="Evaluating datasets", disable=not self.show_progress_bar):
            logger.info(f"Evaluating {evaluator.name}")
            evaluation = evaluator(model, output_path, epoch, steps)
            for k in evaluation:
                if self.truncate_dim:
                    dataset, _, metric = k.split("_", maxsplit=2)
                else:
                    dataset, metric = k.split("_", maxsplit=1)
                if metric not in per_metric_results:
                    per_metric_results[metric] = []
                per_dataset_results[dataset + "_" + metric] = evaluation[k]
                per_metric_results[metric].append(evaluation[k])

        agg_results = {}
        for metric in per_metric_results:
            agg_results[metric] = self.aggregate_fn(per_metric_results[metric])

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

        if not self.primary_metric:
            if self.main_score_function is None:
                score_function = max(
                    [(name, agg_results[f"{name}_ndcg@{max(self.ndcg_at_k)}"]) for name in self.score_function_names],
                    key=lambda x: x[1],
                )[0]
                self.primary_metric = f"{score_function}_ndcg@{max(self.ndcg_at_k)}"
            else:
                self.primary_metric = f"{self.main_score_function.value}_ndcg@{max(self.ndcg_at_k)}"

        avg_queries = np.mean([len(evaluator.queries) for evaluator in self.evaluators])
        avg_corpus = np.mean([len(evaluator.corpus) for evaluator in self.evaluators])
        logger.info(f"Average Queries: {avg_queries}")
        logger.info(f"Average Corpus: {avg_corpus}\n")

        for name in self.score_function_names:
            logger.info(f"Aggregated for Score Function: {name}")
            for k in self.accuracy_at_k:
                logger.info("Accuracy@{}: {:.2f}%".format(k, agg_results[f"{name}_accuracy@{k}"] * 100))

            for k in self.precision_recall_at_k:
                logger.info("Precision@{}: {:.2f}%".format(k, agg_results[f"{name}_precision@{k}"] * 100))
                logger.info("Recall@{}: {:.2f}%".format(k, agg_results[f"{name}_recall@{k}"] * 100))

            for k in self.mrr_at_k:
                logger.info("MRR@{}: {:.4f}".format(k, agg_results[f"{name}_mrr@{k}"]))

            for k in self.ndcg_at_k:
                logger.info("NDCG@{}: {:.4f}".format(k, agg_results[f"{name}_ndcg@{k}"]))

        # TODO: Ensure this primary_metric works as expected, also with bolding the right thing in the model card
        agg_results = self.prefix_name_to_metrics(agg_results, self.name)
        self.store_metrics_in_model_card_data(model, agg_results)

        per_dataset_results.update(agg_results)

        return per_dataset_results

    def _get_human_readable_name(self, dataset_name: DatasetNameType) -> str:
        human_readable_name = f"Nano{dataset_name_to_human_readable[dataset_name.lower()]}"
        if self.truncate_dim is not None:
            human_readable_name += f"_{self.truncate_dim}"
        return human_readable_name

    def _load_dataset(self, dataset_name: DatasetNameType, **ir_evaluator_kwargs) -> InformationRetrievalEvaluator:
        if not is_datasets_available():
            raise ValueError("datasets is not available. Please install it to use the NanoBEIREvaluator.")
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
        return InformationRetrievalEvaluator(
            queries=queries_dict,
            corpus=corpus_dict,
            relevant_docs=qrels_dict,
            name=human_readable_name,
            **ir_evaluator_kwargs,
        )

    def _validate_dataset_names(self):
        if missing_datasets := [
            dataset_name for dataset_name in self.dataset_names if dataset_name.lower() not in dataset_name_to_id
        ]:
            raise ValueError(
                f"Dataset(s) {missing_datasets} not found in the NanoBEIR collection."
                f"Valid dataset names are: {list(dataset_name_to_id.keys())}"
            )

    def _validate_prompts(self):
        error_msg = ""
        if self.query_prompts is not None:
            if isinstance(self.query_prompts, str):
                self.query_prompts = {dataset_name: self.query_prompts for dataset_name in self.dataset_names}
            elif missing_query_prompts := [
                dataset_name for dataset_name in self.dataset_names if dataset_name not in self.query_prompts
            ]:
                error_msg += f"The following datasets are missing query prompts: {missing_query_prompts}\n"

        if self.corpus_prompts is not None:
            if isinstance(self.corpus_prompts, str):
                self.corpus_prompts = {dataset_name: self.corpus_prompts for dataset_name in self.dataset_names}
            elif missing_corpus_prompts := [
                dataset_name for dataset_name in self.dataset_names if dataset_name not in self.corpus_prompts
            ]:
                error_msg += f"The following datasets are missing corpus prompts: {missing_corpus_prompts}\n"

        if error_msg:
            raise ValueError(error_msg.strip())
