from __future__ import annotations

from typing_extensions import deprecated

from sentence_transformers.cross_encoder.evaluation.CEClassificationEvaluator import CEClassificationEvaluator
from sentence_transformers.readers.InputExample import InputExample


@deprecated(
    "This evaluator has been deprecated in favor of the more general CEClassificationEvaluator. "
    "Please use CEClassificationEvaluator instead, which supports both binary and multi-class "
    "evaluation. It accepts approximately the same inputs as this evaluator."
)
class CEF1Evaluator(CEClassificationEvaluator):
    """
    This evaluator has been deprecated in favor of the more general CEClassificationEvaluator.
    """

    @classmethod
    def from_input_examples(cls, examples: list[InputExample], **kwargs):
        sentence_pairs = []
        labels = []

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)
        return cls(sentence_pairs, labels, **kwargs)
