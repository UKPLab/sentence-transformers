import re
from typing import TYPE_CHECKING, Any, Dict, Union

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer


class SentenceEvaluator:
    """
    Base class for all evaluators

    Extend this class and implement __call__ for custom evaluators.
    """

    def __init__(self):
        """
        Base class for all evaluators. Notably, this class introduces the ``greater_is_better`` and ``primary_metric``
        attributes. The former is a boolean indicating whether a higher evaluation score is better, which is used
        for choosing the best checkpoint if ``load_best_model_at_end`` is set to ``True`` in the training arguments.

        The latter is a string indicating the primary metric for the evaluator. This has to be defined whenever
        the evaluator returns a dictionary of metrics, and the primary metric is the key pointing to the primary
        metric, i.e. the one that is used for model selection and/or logging.
        """
        self.greater_is_better = True
        self.primary_metric = None

    def __call__(
        self, model: "SentenceTransformer", output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> Union[float, Dict[str, float]]:
        """
        This is called during training to evaluate the model.
        It returns a score for the evaluation with a higher score indicating a better result.

        Args:
            model: the model to evaluate
            output_path: path where predictions and metrics are written
                to
            epoch: the epoch where the evaluation takes place. This is
                used for the file prefixes. If this is -1, then we
                assume evaluation on test data.
            steps: the steps in the current epoch at time of the
                evaluation. This is used for the file prefixes. If this
                is -1, then we assume evaluation at the end of the
                epoch.

        Returns:
            Either a score for the evaluation with a higher score
            indicating a better result, or a dictionary with scores. If
            the latter is chosen, then `evaluator.primary_metric` must
            be defined
        """
        pass

    def prefix_name_to_metrics(self, metrics: Dict[str, float], name: str):
        if not name:
            return metrics
        metrics = {name + "_" + key: value for key, value in metrics.items()}
        if hasattr(self, "primary_metric") and not self.primary_metric.startswith(name + "_"):
            self.primary_metric = name + "_" + self.primary_metric
        return metrics

    def store_metrics_in_model_card_data(self, model: "SentenceTransformer", metrics: Dict[str, Any]) -> None:
        model.model_card_data.set_evaluation_metrics(self, metrics)

    @property
    def description(self) -> str:
        """
        Returns a human-readable description of the evaluator: BinaryClassificationEvaluator -> Binary Classification

        1. Remove "Evaluator" from the class name
        2. Add a space before every capital letter
        """
        class_name = self.__class__.__name__
        try:
            index = class_name.index("Evaluator")
            class_name = class_name[:index]
        except IndexError:
            pass

        return re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", class_name)
