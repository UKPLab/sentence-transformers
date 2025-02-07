from __future__ import annotations

# TODO: Consider renaming all evaluators to CrossEncoder..., e.g. CrossEncoderNanoBEIREvaluator, CrossEncoderClassificationEvaluator, etc.
from .CEBinaryAccuracyEvaluator import CEBinaryAccuracyEvaluator
from .CEBinaryClassificationEvaluator import CEBinaryClassificationEvaluator
from .CEClassificationEvaluator import CEClassificationEvaluator
from .CECorrelationEvaluator import CECorrelationEvaluator
from .CEF1Evaluator import CEF1Evaluator
from .CENanoBEIREvaluator import CENanoBEIREvaluator
from .CERerankingEvaluator import CERerankingEvaluator
from .CESoftmaxAccuracyEvaluator import CESoftmaxAccuracyEvaluator

__all__ = [
    "CEClassificationEvaluator",
    "CECorrelationEvaluator",
    "CERerankingEvaluator",
    "CENanoBEIREvaluator",
    "CEBinaryAccuracyEvaluator",  # Deprecated
    "CEBinaryClassificationEvaluator",  # Deprecated
    "CEF1Evaluator",  # Deprecated
    "CESoftmaxAccuracyEvaluator",  # Deprecated
]
