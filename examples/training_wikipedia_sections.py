"""
This script trains sentence transformers with a triplet loss function.

As corpus, we use the wikipedia sections dataset that was describd by Dor et al., 2018, Learning Thematic Similarity Metric Using Triplet Networks.

See docs/pretrained-models/wikipedia-sections-modesl.md for further details.

You can get the dataset by running examples/datasets/get_data.py
"""

from sentence_transformers import SentenceTransformer,  LossFunction, SentenceTransformerConfig, TrainConfig, TripletMetric, TripletEvaluator,  SentencesDataset, LoggingHandler
from torch.utils.data import DataLoader
from sentence_transformers.dataset_readers import TripletReader

import csv
import logging



logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])



### Create a torch.DataLoader that passes training batch instances to our model
train_batch_size = 16
triplet_reader = TripletReader('datasets/wikipedia-sections-triplets', s1_col_idx=1, s2_col_idx=2, s3_col_idx=3, delimiter=',', quoting=csv.QUOTE_MINIMAL, has_header=True)

### Configure sentence transformers for training and train on the provided dataset
transformer_config = SentenceTransformerConfig(model='sentence_transformers.models.BERT',
                                               tokenizer_model='bert-base-uncased',
                                               do_lower_case=True,
                                               max_seq_length=128,
                                               loss_function=LossFunction.TRIPLET_LOSS,
                                               triplet_margin=0.2,
                                               triplet_metric=TripletMetric.EUCLIDEAN)

embedder = SentenceTransformer(sentence_transformer_config=transformer_config)


logging.info("Read Triplet train dataset")
train_data = SentencesDataset(examples=triplet_reader.get_examples('train.csv', 100), model=embedder)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size, collate_fn=embedder.encoder.smart_batching_collate)


logging.info("Read Wikipedia Triplet dev dataset")
dev_data = SentencesDataset(examples=triplet_reader.get_examples('validation.csv', 1000), model=embedder)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size, collate_fn=embedder.encoder.smart_batching_collate)
evaluator = TripletEvaluator(dev_dataloader)


output_path = "output/bert-base-wikipedia-sections-mean-tokens"

num_epochs = 1
warmup_steps = int(len(train_data)*num_epochs/train_batch_size/10) #10% of train data
train_config = TrainConfig(learning_rate=2e-5,
                           weight_decay=0.01,
                           epochs=num_epochs,
                           evaluation_steps=1000,
                           output_path=output_path,
                           save_best_model=True,
                           evaluator=evaluator,
                           warmup_steps=warmup_steps)


embedder.train(dataloader=train_dataloader, train_config=train_config)

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

embedder = SentenceTransformer(output_path)
test_data =SentencesDataset(examples=triplet_reader.get_examples('test.csv'), model=embedder)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=train_batch_size, collate_fn=embedder.encoder.smart_batching_collate)
evaluator = TripletEvaluator(test_dataloader)

embedder.evaluate(evaluator)

