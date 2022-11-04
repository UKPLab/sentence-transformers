
from sentence_transformers import SentenceTransformer, LoggingHandler
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from mteb import MTEB
from mteb.tasks import AmazonReviewsClassification



model_name_pre = "nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large"
model_beforetraining = SentenceTransformer(model_name_pre)
print(model_beforetraining.encode("hello"))
evaluation = MTEB(tasks=["AmazonReviewsClassification"])
results = evaluation.run(model_beforetraining, output_folder=f"mteb_results/{model_name_pre}")

#model trained on ted2020
model_name = "./model_mMiniLMv2-L12-H384-distilled-from-XLMR-Large_ted2020_pairs"
model = SentenceTransformer(model_name)
print(model.encode("hello"))
evaluation = MTEB(tasks=["AmazonReviewsClassification"])
results = evaluation.run(model, output_folder=f"mteb_results/{model_name}")

model_name_post = "./model_model_mMiniLMv2-L12-H384-distilled-from-XLMR-Large_ted2020_pairs_globalvoices_pairs"
model_posttraining = SentenceTransformer(model_name_post)
print(model_posttraining.encode("hello"))
evaluation = MTEB(tasks=["AmazonReviewsClassification"])
results = evaluation.run(model_posttraining, output_folder=f"mteb_results/{model_name_post}")