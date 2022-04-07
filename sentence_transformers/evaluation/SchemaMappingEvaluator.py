import logging
import os
from typing import List

import numpy as np
from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction
from sentence_transformers.readers import InputExample
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances  # noqa: F401

from semantic import service_internal
from semantic.calibrate_rank_scaler import calibrate_rank_scaler
from semantic.sbert import SBert
from semantic.scripts.scaler import RankScaler
from semantic.scripts.word_tokenizer import WordTokenizer
from semantic.test import plot_precision_recall

logger = logging.getLogger(__name__)


class SchemaMappingEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(
        self,
        sentences1: List[str],
        sentences2: List[str],
        scores: List[float],
        test_schemas: List[dict] = None,
        opt: dict = None,
        batch_size: int = 16,
        main_similarity: SimilarityFunction = None,
        name: str = "",
        show_progress_bar: bool = False,
        write_csv: bool = True,
    ):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param sentences1:  List with the first sentence in a pair
        :param sentences2: List with the second sentence in a pair
        :param scores: Similarity score between sentences1[i] and sentences2[i]
        :param write_csv: Write results to a CSV file
        """
        assert opt is not None
        assert test_schemas is not None
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        self.write_csv = write_csv
        self.opt = opt
        self.test_schemas = test_schemas
        self.word_tokenizer = WordTokenizer(opt["model"]["tokenizer"])

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.scores)

        self.main_similarity = main_similarity
        self.name = name

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
        self.show_progress_bar = show_progress_bar

        self.csv_file = "similarity_evaluation" + ("_" + name if name else "") + "_results.csv"
        self.csv_headers = [
            "epoch",
            "steps",
            "cosine_pearson",
            "cosine_spearman",
            "euclidean_pearson",
            "euclidean_spearman",
            "manhattan_pearson",
            "manhattan_spearman",
            "dot_pearson",
            "dot_spearman",
        ]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        return cls(sentences1, sentences2, scores, **kwargs)

    def eval_schemas(self, model):
        old_threshold = self.opt["model"]["threshold"]
        self.opt["model"]["threshold"] = 0

        # prepare service internal for predict
        service_internal.sbert_model = SBert(self.opt, model=model)
        service_internal.word_tok = self.word_tokenizer
        service_internal.opt = self.opt
        if self.opt["model"]["rank_calibration"]:
            calibrate_rank_scaler(self.opt, service_internal.sbert_model, service_internal.word_tok)
            service_internal.scaler_words = RankScaler(os.path.join(self.opt["model"]["model_path"], "scaler_words.npz"))
            service_internal.scaler_phrase = RankScaler(os.path.join(self.opt["model"]["model_path"], "scaler_phrases.npz"))

        # predict schema matching
        target_count = 0
        pred_thresholds = []
        for schema in self.test_schemas:

            # prepare an example
            request = {"source": [], "target": [], "user_mappings": []}
            target_mapping = {}
            distractor_src, distractor_target = [], []
            for txt1, txt2, label in schema:
                request["source"].append({"text": txt1})
                request["target"].append({"text": txt2})
                if label:
                    target_mapping[txt1] = txt2
                else:
                    distractor_src.append(txt1)
                    distractor_target.append(txt2)

            if not target_mapping:
                continue
            target_count += len(target_mapping)
            rez, _ = service_internal.predict({"request": {"data": request}}, verbose=False)

            # measure score
            pred_thr = []
            for r in rez["predictions"]:
                rez_src, rez_targ, rez_score = r
                # ignore distractor to distractor mappings
                if rez_src in distractor_src and rez_targ in distractor_target:
                    continue
                pred_ok = int(target_mapping.get(rez_src) == rez_targ)
                pred_thr.append((pred_ok, rez_score))
                print(f"{'OK ' if pred_ok else 'BAD'} | {rez_score:.2f} | src={rez_src:30}  | targ={str(target_mapping.get(rez_src)):20} | pred={rez_targ}")
            pred_thresholds.append(pred_thr)

        recall_at_precision_90, report = plot_precision_recall(self.opt, pred_thresholds, target_count, plot_curves=False)

        self.opt["model"]["threshold"] = old_threshold

        return recall_at_precision_90

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("SchemaMappingEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        embeddings1 = model.encode(self.sentences1, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        embeddings2 = model.encode(self.sentences2, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        labels = self.scores

        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        average_acc = np.average([bool(ll >= 0.5) == bool(ss >= 0.5) for ll, ss in zip(labels, cosine_scores)])

        recall_at_precision_90 = self.eval_schemas(model)

        logger.info(f"Average accuracy  :\t{average_acc:.4f} recall@precision_90={recall_at_precision_90:.4f}")
        return recall_at_precision_90
