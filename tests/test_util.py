from sentence_transformers import util, SentenceTransformer
import unittest
import numpy as np
import sklearn
import torch

class UtilTest(unittest.TestCase):

    def test_normalize_embeddings(self):
        """Tests the correct computation of util.normalize_embeddings"""
        embedding_size = 100
        a = torch.tensor(np.random.randn(50, embedding_size))
        a_norm = util.normalize_embeddings(a)

        for embedding in a_norm:
            assert len(embedding) == embedding_size
            emb_norm = torch.norm(embedding)
            assert abs(emb_norm.item() - 1) < 0.0001


    def test_pytorch_cos_sim(self):
        """Tests the correct computation of util.pytorch_cos_scores"""
        a = np.random.randn(50, 100)
        b = np.random.randn(50, 100)

        sklearn_pairwise = sklearn.metrics.pairwise.cosine_similarity(a, b)
        pytorch_cos_scores = util.pytorch_cos_sim(a, b).numpy()
        for i in range(len(sklearn_pairwise)):
            for j in range(len(sklearn_pairwise[i])):
                assert abs(sklearn_pairwise[i][j] - pytorch_cos_scores[i][j]) < 0.001


    def test_semantic_search(self):
        """Tests util.semantic_search function"""
        num_queries = 20
        num_k = 10

        doc_emb = torch.tensor(np.random.randn(1000, 100))
        q_emb = torch.tensor(np.random.randn(num_queries, 100))
        hits = util.semantic_search(q_emb, doc_emb, top_k=num_k, query_chunk_size=5, corpus_chunk_size=17)
        assert len(hits) == num_queries
        assert len(hits[0]) == num_k

        #Sanity Check of the results
        cos_scores = util.pytorch_cos_sim(q_emb, doc_emb)
        cos_scores_values, cos_scores_idx = cos_scores.topk(num_k)
        cos_scores_values = cos_scores_values.cpu().tolist()
        cos_scores_idx = cos_scores_idx.cpu().tolist()

        for qid in range(num_queries):
            for hit_num in range(num_k):
                assert hits[qid][hit_num]['corpus_id'] == cos_scores_idx[qid][hit_num]
                assert np.abs(hits[qid][hit_num]['score'] - cos_scores_values[qid][hit_num]) < 0.001

    def test_paraphrase_mining(self):
        model = SentenceTransformer('paraphrase-distilroberta-base-v1')
        sentences = [
            "This is a test", "This is a test!",
            "The cat sits on mat", "The cat sits on the mat", "On the mat a cat sits",
            "A man eats pasta", "A woman eats pasta", "A man eats spaghetti"
        ]
        duplicates = util.paraphrase_mining(model, sentences)

        for score, a, b in duplicates:
            if score > 0.5:
                assert (a,b) in [(0,1), (2,3), (2,4), (3,4), (5,6), (5,7), (6,7)]



if "__main__" == __name__:
    unittest.main()