from . import SentenceEvaluator
from typing import Iterable


class SequentialEvaluator(SentenceEvaluator):
    """
    This evaluator allows that multiple sub-evaluators are passed. When the model is evaluated,
    the data is passed sequentially to all sub-evaluators.

    All scores are passed to 'main_score_function', which derives one final score value
    """

    def __init__(
        self,
        evaluators: Iterable[SentenceEvaluator],
        main_score_function=lambda scores: scores[-1],
        scoring_function_idx: int = -1,
    ):
        self.evaluators = evaluators
        self.main_score_function = main_score_function
        self.scoring_function_idx = scoring_function_idx

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        scores = []
        for evaluator in self.evaluators:
            scores.append(evaluator(model, output_path, epoch, steps))
            if self.scoring_function_idx == -1 and hasattr(evaluator, "best_scoring_function"):
                self.best_scoring_function = evaluator.best_scoring_function

        if self.scoring_function_idx != -1:
            self.best_scoring_function = self.evaluators[self.scoring_function_idx].best_scoring_function

        return self.main_score_function(scores)
