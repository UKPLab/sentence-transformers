from . import SentenceEvaluator
from typing import Iterable

class SequentialEvaluator(SentenceEvaluator):
    """
    This evaluator allows that multiple sub-evaluators are passed. When the model is evaluated,
    the data is passed sequentially to all sub-evaluators.

    The score from the last sub-evaluator will be used as the main score for the best model decision.
    """
    def __init__(self, evaluators: Iterable[SentenceEvaluator]):
        self.evaluators = evaluators

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        for evaluator in self.evaluators:
            main_score = evaluator(model, output_path, epoch, steps)

        return main_score
