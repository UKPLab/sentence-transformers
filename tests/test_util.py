from sentence_transformers import  util
import unittest
import numpy as np
import sklearn

class UtilTest(unittest.TestCase):

    def test_pytorch_cos_sim(self):
        """Tests the correct computation of util.pytorch_cos_scores"""
        a = np.random.randn(50, 100)
        b = np.random.randn(50, 100)

        sklearn_pairwise = sklearn.metrics.pairwise.cosine_similarity(a, b)
        pytorch_cos_scores = util.pytorch_cos_sim(a, b).numpy()
        for i in range(len(sklearn_pairwise)):
            for j in range(len(sklearn_pairwise[i])):
                assert abs(sklearn_pairwise[i][j] - pytorch_cos_scores[i][j]) < 0.001

if "__main__" == __name__:
    unittest.main()