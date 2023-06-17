"""
Computes embeddings
"""


import unittest
from sentence_transformers import CT2SentenceTransformer, SentenceTransformer
import numpy as np
import torch
from timeit import default_timer as timer
import random


class CT2ComputeEmbeddingsTest(unittest.TestCase):
    def setUp(
        self,
        compute_type="default",
        device="cpu",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = CT2SentenceTransformer(
            model_name, compute_type=compute_type, device=device
        )
        self.default_model = SentenceTransformer(model_name, device=device)
        self.abs_tol = 5e-3
        self.embed_dim = 384
        self.do_speed_test = bool(compute_type == "default")
        if "float16" in compute_type:
            self.abs_tol = 5e-3
        if "int8" in compute_type:
            self.abs_tol = 5e-2

    def test_encode_token_embeddings(self):
        """
        Test that encode(output_value='token_embeddings') works
        :return:
        """
        sent = [
            "Hello Word, a test sentence",
            "Here comes another sentence",
            "My final sentence",
            "Sentences",
            "Sentence five five five five five five five",
        ]
        emb = self.model.encode(sent, output_value="token_embeddings", batch_size=2)
        assert len(emb) == len(sent)
        for s, e in zip(sent, emb):
            assert len(self.model.tokenize([s])["input_ids"][0]) == e.shape[0]

    def test_encode_single_sentences(self):
        for sentence, shape in [
            ("Hello Word, a test sentence", (self.embed_dim,)),
            (["Hello Word, a test sentence"], (1, self.embed_dim)),
            (
                [
                    "Hello Word, a test sentence",
                    "Here comes another sentence",
                    "My final sentence",
                ],
                (3, self.embed_dim),
            ),
        ]:
            emb = self.model.encode(sentence)
            emb_default = self.default_model.encode(sentence)
            assert emb.shape == shape
            assert np.abs(emb_default - emb).mean() < self.abs_tol

    def test_encode_normalize(self):
        for sentence, shape in [
            (["Hello Word, a test sentence"], (1, self.embed_dim)),
            (
                [
                    "Hello Word, a test sentence",
                    "Here comes another sentence",
                    "My final sentence",
                ],
                (3, self.embed_dim),
            ),
        ]:
            emb = self.model.encode(sentence, normalize_embeddings=True)
            emb_default = self.default_model.encode(sentence, normalize_embeddings=True)
            assert emb.shape == shape
            assert np.abs(emb_default - emb).mean() < self.abs_tol
            for norm in np.linalg.norm(emb, axis=1):
                assert abs(norm - 1) < self.abs_tol

    def test_encode_tuple_sentences(self):
        # Input a sentence tuple
        for sentence, shape in [
            ([("Hello Word, a test sentence", "Second input for model")], (1, self.embed_dim)),
            (
                [
                    ("Hello Word, a test sentence", "Second input for model"),
                    ("My second tuple", "With two inputs"),
                    ("Final tuple", "final test"),
                ],
                (3, self.embed_dim),
            ),
        ]:
            emb = self.model.encode(sentence)
            emb_default = self.default_model.encode(sentence)
            assert emb.shape == shape
            assert np.abs(emb_default - emb).mean() < self.abs_tol

    def test_encoding_latency(self):
        # asserting that latency of ct2 is worst case 2x of torch
        # usually its faster, between 0.9x and 0.5x
        if self.do_speed_test:
            sentence = [
                "".join(
                    random.choice(["one ", "two ", "hi "]) for length in range(batch_idx+1)
                )
                for batch_idx in range(64)
            ]

            def timing(model):
                times = []
                model.encode("warmup")
                for _ in range(3):
                    start = timer()
                    model.encode(sentence, batch_size=16)
                    end = timer()
                    times.append(end - start)
                return np.median(times)

            time_ct2 = timing(self.model)
            time_default = timing(self.default_model)

            assert float(time_ct2 / time_default) < 2
            
            

class int8_cpu(CT2ComputeEmbeddingsTest):
    def setUp(self):
        super().setUp(compute_type="int8", device="cpu")


class default_cpu_e5_small(CT2ComputeEmbeddingsTest):
    # testing feature extraction model from HF.
    def setUp(self):
        super().setUp(device="cpu", model_name="intfloat/e5-small")


if torch.cuda.is_available():

    class default_cuda(CT2ComputeEmbeddingsTest):
        def setUp(self):
            super().setUp(device="cuda")

    class int8float16_cuda(CT2ComputeEmbeddingsTest):
        def setUp(self):
            super().setUp(compute_type="int8_float16", device="cuda")

    class int8_cuda(CT2ComputeEmbeddingsTest):
        def setUp(self):
            super().setUp(compute_type="int8", device="cuda")

    class float16_cuda(CT2ComputeEmbeddingsTest):
        def setUp(self):
            super().setUp(compute_type="float16", device="cuda")

    class float32_cuda(CT2ComputeEmbeddingsTest):
        def setUp(self):
            super().setUp(compute_type="float32", device="cuda")
            
    