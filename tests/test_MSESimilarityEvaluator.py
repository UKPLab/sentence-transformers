"""
Tests the correct computation of evaluation scores from MSESimilarityEvaluator
"""

from sentence_transformers import (
    SentenceTransformer,
    evaluation,
)

PRITAMDEKA_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
def test_MSESimilarityEvaluator(model : SentenceTransformer = SentenceTransformer(PRITAMDEKA_MODEL)) -> None:
    """Tests that the MSESimilarityEvaluator can be loaded correctly"""
    s1 = ["My first sentence", "Another pair"]
    s2 = ["My second sentence", "Unrelated sentence"]
    gold_labels = [0.8, 0.3]
    evaluator=evaluation.MSESimilarityEvaluator(s1, s2, gold_labels, main_similarity=evaluation.SimilarityFunction.COSINE)
    metrics = evaluator(model)
    assert metrics > 0