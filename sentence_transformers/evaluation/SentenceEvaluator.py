from typing import Dict, Union


class SentenceEvaluator:
    """
    Base class for all evaluators

    Extend this class and implement __call__ for custom evaluators.
    """

    def __init__(self):
        self.greater_is_better = True

    def __call__(
        self, model, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> Union[float, Dict[str, float]]:
        """
        This is called during training to evaluate the model.
        It returns a score for the evaluation with a higher score indicating a better result.

        :param model:
            the model to evaluate
        :param output_path:
            path where predictions and metrics are written to
        :param epoch
            the epoch where the evaluation takes place.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation on test data.
        :param steps
            the steps in the current epoch at time of the evaluation.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation at the end of the epoch.
        :return: Either a score for the evaluation with a higher score indicating a better result,
            or a dictionary with scores. If the latter is chosen, then `evaluator.primary_metric`
            must be defined
        """
        pass
