"""
Tests the correct computation of evaluation scores from BinaryClassificationEvaluator
"""
from sentence_transformers import SentenceTransformer, evaluation
import unittest
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

class BinaryClassificationEvaluatorTest(unittest.TestCase):

    def test_find_best_f1_and_threshold(self):
        """Tests that the F1 score for the computed threshold is correct"""
        y_true = np.random.randint(0, 2, 1000)
        y_pred_cosine = np.random.randn(1000)
        best_f1, best_precision, best_recall, threshold = evaluation.BinaryClassificationEvaluator.find_best_f1_and_threshold(y_pred_cosine, y_true, high_score_more_similar=True)
        y_pred_labels = [1 if pred >= threshold else 0 for pred in y_pred_cosine]
        sklearn_f1score = f1_score(y_true, y_pred_labels)
        assert np.abs(best_f1 - sklearn_f1score) < 1e-6


    def test_find_best_accuracy_and_threshold(self):
        """Tests that the Acc score for the computed threshold is correct"""
        y_true = np.random.randint(0, 2, 1000)
        y_pred_cosine = np.random.randn(1000)
        max_acc, threshold = evaluation.BinaryClassificationEvaluator.find_best_acc_and_threshold(y_pred_cosine, y_true, high_score_more_similar=True)
        y_pred_labels = [1 if pred >= threshold else 0 for pred in y_pred_cosine]
        sklearn_acc = accuracy_score(y_true, y_pred_labels)
        assert np.abs(max_acc - sklearn_acc) < 1e-6
