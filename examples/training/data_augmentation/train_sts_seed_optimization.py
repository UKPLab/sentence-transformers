"""
This script is identical to examples/training/sts/training_stsbenchmark.py with seed optimization.
We apply early stopping and evaluate the models over the dev set, to find out the best perfoming seeds.

For more details refer to -
Fine-Tuning Pretrained Language Models:
Weight Initializations, Data Orders, and Early Stopping by Dodge et al. 2020 
https://arxiv.org/pdf/2002.06305.pdf

Why Seed Optimization?
Dodge et al. (2020) show a high dependence on the random seed for transformer based models like BERT, 
as it converges to different minima that generalize differently to unseen data. This is especially the 
case for small training datasets. 

Citation: https://arxiv.org/abs/2010.08240

Usage:
python train_sts_seed_optimization.py

OR
python train_sts_seed_optimization.py pretrained_transformer_model_name seed_count stop_after

python ttrain_sts_seed_optimization.py bert-base-uncased 10 0.3
"""
from torch.utils.data import DataLoader
import math
import torch
import random
import numpy as np
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import sys
import os
import gzip
import csv

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout



#Check if dataset exsist. If not, download and extract  it
sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)



#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'
seed_count = int(sys.argv[2]) if len(sys.argv) > 2 else 10
stop_after = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3

logging.info("Train and Evaluate: {} Random Seeds".format(seed_count))

for seed in range(seed_count):

    # Setting seed for all random initializations
    logging.info("##### Seed {} #####".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Read the dataset
    train_batch_size = 16
    num_epochs = 1
    model_save_path = 'output/bi-encoder/training_stsbenchmark_'+ model_name + '/seed-'+ str(seed)

    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                pooling_mode_mean_tokens=True,
                                pooling_mode_cls_token=False,
                                pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Convert the dataset to a DataLoader ready for training
    logging.info("Read STSbenchmark train dataset")

    train_samples = []
    dev_samples = []
    test_samples = []
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
            inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

            if row['split'] == 'dev':
                dev_samples.append(inp_example)
            elif row['split'] == 'test':
                test_samples.append(inp_example)
            else:
                train_samples.append(inp_example)


    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)


    logging.info("Read STSbenchmark dev dataset")
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')


    # Configure the training. We skip evaluation in this example
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
    
    # Stopping and Evaluating after 30% of training data (less than 1 epoch)
    # We find from (Dodge et al.) that 20-30% is often ideal for convergence of random seed
    steps_per_epoch = math.ceil( len(train_dataset) / train_batch_size * stop_after ) 
    
    logging.info("Warmup-steps: {}".format(warmup_steps))

    logging.info("Early-stopping: {}% of the training-data".format(int(stop_after*100)))


    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=model_save_path)
