from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
from tqdm import tqdm

from sentence_transformers.cross_encoder.evaluation.CERerankingEvaluator import CERerankingEvaluator
from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.util import is_datasets_available

if TYPE_CHECKING:
    from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder

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
    "climatefever": "tomaarsen/NanoClimateFEVER-bm25",
    "dbpedia": "tomaarsen/NanoDBPedia-bm25",
    "fever": "tomaarsen/NanoFEVER-bm25",
    "fiqa2018": "tomaarsen/NanoFiQA2018-bm25",
    "hotpotqa": "tomaarsen/NanoHotpotQA-bm25",
    "msmarco": "tomaarsen/NanoMSMARCO-bm25",
    "nfcorpus": "tomaarsen/NanoNFCorpus-bm25",
    "nq": "tomaarsen/NanoNQ-bm25",
    "quoraretrieval": "tomaarsen/NanoQuoraRetrieval-bm25",
    "scidocs": "tomaarsen/NanoSCIDOCS-bm25",
    "arguana": "tomaarsen/NanoArguAna-bm25",
    "scifact": "tomaarsen/NanoSciFact-bm25",
    "touche2020": "tomaarsen/NanoTouche2020-bm25",
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


class CENanoBEIREvaluator(SentenceEvaluator):
    def __init__(
        self,
        dataset_names: list[DatasetNameType] | None = None,
        rerank_k: int = 100,
        at_k: int = 10,
        show_progress_bar: bool = False,
        batch_size: int = 32,
        write_csv: bool = True,
        aggregate_fn: Callable[[list[float]], float] = np.mean,
        aggregate_key: str = "mean",
    ):
        super().__init__()
        if dataset_names is None:
            dataset_names = list(
                dataset_name_to_id.keys()
            )  # TODO: Consider removing arguana and touche2020 from the default
        self.dataset_names = dataset_names
        self.rerank_k = rerank_k
        self.at_k = at_k
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.write_csv = write_csv
        self.aggregate_fn = aggregate_fn
        self.aggregate_key = aggregate_key

        self.name = f"NanoBEIR_{self.aggregate_key}"

        self._validate_dataset_names()

        reranking_kwargs = {
            "at_k": self.at_k,
            "negatives_are_ranked": True,
            "show_progress_bar": self.show_progress_bar,
            "batch_size": self.batch_size,
            "write_csv": self.write_csv,
        }

        self.evaluators = [
            self._load_dataset(name, **reranking_kwargs)
            for name in tqdm(self.dataset_names, desc="Loading NanoBEIR datasets", leave=False)
        ]

        # TODO: This is not used
        self.csv_file: str = f"NanoBEIR_evaluation_{aggregate_key}_results.csv"
        self.csv_headers = ["epoch", "steps"]

        self.primary_metric = "ndcg@10"  # TODO: Move this elsewhere, set it dynamically, etc.

    def __call__(
        self, model: CrossEncoder, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs
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
        logger.info(f"NanoBEIR Evaluation of the model on {self.dataset_names} dataset{out_txt}:")

        for evaluator in tqdm(self.evaluators, desc="Evaluating datasets", disable=not self.show_progress_bar):
            logger.info(f"Evaluating {evaluator.name}")
            evaluation = evaluator(model, output_path, epoch, steps)
            for k in evaluation:
                dataset, metric = k.split("_", maxsplit=1)
                if metric not in per_metric_results:
                    per_metric_results[metric] = []
                per_dataset_results[dataset + "_" + metric] = evaluation[k]
                per_metric_results[metric].append(evaluation[k])
            logger.info("")

        agg_results = {}
        for metric in per_metric_results:
            agg_results[metric] = self.aggregate_fn(per_metric_results[metric])

        logger.info("CENanoBEIREvaluator: Aggregated Results:")
        logger.info("\t\tBase  -> Reranked")
        logger.info(f"MAP:\t{agg_results['base_map'] * 100:.2f} -> {agg_results['map'] * 100:.2f}")
        logger.info(
            f"MRR@{self.at_k}:\t{agg_results[f'base_mrr@{self.at_k}'] * 100:.2f} -> {agg_results[f'mrr@{self.at_k}'] * 100:.2f}"
        )
        logger.info(
            f"NDCG@{self.at_k}:\t{agg_results[f'base_ndcg@{self.at_k}'] * 100:.2f} -> {agg_results[f'ndcg@{self.at_k}'] * 100:.2f}"
        )

        model_card_metrics = {
            "map": f"{agg_results['map']:.4f} ({agg_results['map'] - agg_results['base_map']:+.4f})",
            f"mrr@{self.at_k}": f"{agg_results[f'mrr@{self.at_k}']:.4f} ({agg_results[f'mrr@{self.at_k}'] - agg_results[f'base_mrr@{self.at_k}']:+.4f})",
            f"ndcg@{self.at_k}": f"{agg_results[f'ndcg@{self.at_k}']:.4f} ({agg_results[f'ndcg@{self.at_k}'] - agg_results[f'base_ndcg@{self.at_k}']:+.4f})",
        }
        model_card_metrics = self.prefix_name_to_metrics(model_card_metrics, self.name)
        self.store_metrics_in_model_card_data(model, model_card_metrics, epoch, steps)

        agg_results = self.prefix_name_to_metrics(agg_results, self.name)
        per_dataset_results.update(agg_results)

        return per_dataset_results

    def _get_human_readable_name(self, dataset_name: DatasetNameType) -> str:
        human_readable_name = f"Nano{dataset_name_to_human_readable[dataset_name.lower()]}"
        return human_readable_name

    def _load_dataset(self, dataset_name: DatasetNameType, **ir_evaluator_kwargs) -> CERerankingEvaluator:
        if not is_datasets_available():
            raise ValueError(
                "datasets is not available. Please install it to use the CENanoBEIREvaluator via `pip install datasets`."
            )
        from datasets import load_dataset

        dataset_path = dataset_name_to_id[dataset_name.lower()]
        corpus = load_dataset(dataset_path, "corpus", split="train")
        corpus_mapping = dict(zip(corpus["_id"], corpus["text"]))
        queries = load_dataset(dataset_path, "queries", split="train")
        query_mapping = dict(zip(queries["_id"], queries["text"]))
        relevance = load_dataset(dataset_path, "relevance", split="train")

        def mapper(sample, corpus_mapping: dict[str, str], query_mapping: dict[str, str], rerank_k: int):
            query = query_mapping[sample["query-id"]]
            positives = [corpus_mapping[positive] for positive in sample["positive-corpus-ids"]]
            negatives = [corpus_mapping[negative] for negative in sample["bm25-ranked-ids"][:rerank_k]]
            return {
                "query": query,
                "positive": positives,
                "negative": negatives,
            }

        relevance = relevance.map(
            mapper,
            fn_kwargs={"corpus_mapping": corpus_mapping, "query_mapping": query_mapping, "rerank_k": self.rerank_k},
        )

        human_readable_name = self._get_human_readable_name(dataset_name)
        return CERerankingEvaluator(
            samples=list(relevance),
            name=human_readable_name,
            **ir_evaluator_kwargs,
        )

    def _validate_dataset_names(self):
        if len(self.dataset_names) == 0:
            raise ValueError("dataset_names cannot be empty. Use None to evaluate on all datasets.")
        if missing_datasets := [
            dataset_name for dataset_name in self.dataset_names if dataset_name.lower() not in dataset_name_to_id
        ]:
            raise ValueError(
                f"Dataset(s) {missing_datasets} not found in the NanoBEIR collection. "
                f"Valid dataset names are: {list(dataset_name_to_id.keys())}"
            )
