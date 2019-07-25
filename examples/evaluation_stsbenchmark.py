"""
This examples loads a pre-trained model
"""
from torch.utils.data import DataLoader
from sentence_transformers import  SentenceTransformer, LossFunction, TrainConfig, SentencesDataset, LoggingHandler, EmbeddingSimilarityEvaluator, EmbeddingSimilarity
from sentence_transformers.dataset_readers import STSDataReader
import numpy as np
import logging


#### Just some code to print debug information to stdout
np.set_printoptions(threshold=100)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout



# Load Sentence model (based on BERT) from URL
model = SentenceTransformer('bert-base-nli-mean')


sts_reader = STSDataReader('datasets/stsbenchmark')

test_data = SentencesDataset(examples=sts_reader.get_examples("sts-test.csv"), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=8, collate_fn=model.smart_batching_collate())
evaluator = EmbeddingSimilarityEvaluator(test_dataloader, EmbeddingSimilarity.COSINE)

model.evaluate(evaluator)
