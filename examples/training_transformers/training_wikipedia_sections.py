"""
This script trains sentence transformers with a triplet loss function.

As corpus, we use the wikipedia sections dataset that was describd by Dor et al., 2018, Learning Thematic Similarity Metric Using Triplet Networks.

See docs/pretrained-models/wikipedia-sections-modesl.md for further details.

You can get the dataset by running examples/datasets/get_data.py
"""

from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models
from torch.utils.data import DataLoader
from sentence_transformers.readers import TripletReader
from sentence_transformers.evaluation import TripletEvaluator
from datetime import datetime

import csv
import logging



logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = 'bert-base-uncased'


### Create a torch.DataLoader that passes training batch instances to our model
train_batch_size = 16
triplet_reader = TripletReader('../datasets/wikipedia-sections-triplets', s1_col_idx=1, s2_col_idx=2, s3_col_idx=3, delimiter=',', quoting=csv.QUOTE_MINIMAL, has_header=True)
output_path = "output/training-wikipedia-sections-"+model_name+"-"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
num_epochs = 1


### Configure sentence transformers for training and train on the provided dataset
# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


logging.info("Read Triplet train dataset")
train_dataset = SentencesDataset(examples=triplet_reader.get_examples('train.csv', max_examples=100000), model=model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.TripletLoss(model=model)

logging.info("Read Wikipedia Triplet dev dataset")
evaluator = TripletEvaluator.from_input_examples(triplet_reader.get_examples('validation.csv', 1000), name='dev')


warmup_steps = int(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=output_path)

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(output_path)
test_evaluator = TripletEvaluator.from_input_examples(triplet_reader.get_examples('test.csv'), name='test')
test_evaluator(model, output_path=output_path)

