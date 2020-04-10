"""
This is an example how to train SentenceTransformers in a multi-task setup.

The system trains BERT on the AllNLI and on the STSbenchmark dataset.
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
import logging
from datetime import datetime

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the dataset
model_name = 'bert-base-uncased'
batch_size = 16
nli_reader = NLIDataReader('datasets/AllNLI')
sts_reader = STSDataReader('datasets/stsbenchmark')
train_num_labels = nli_reader.get_num_labels()
model_save_path = 'output/training_multi-task_'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



# Use BERT for mapping tokens to embeddings
word_embedding_model = models.BERT(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# Convert the dataset to a DataLoader ready for training
logging.info("Read AllNLI train dataset")
train_data_nli = SentencesDataset(nli_reader.get_examples('train.gz'), model=model)
train_dataloader_nli = DataLoader(train_data_nli, shuffle=True, batch_size=batch_size)
train_loss_nli = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)

logging.info("Read STSbenchmark train dataset")
train_data_sts = SentencesDataset(sts_reader.get_examples('sts-train.csv'), model=model)
train_dataloader_sts = DataLoader(train_data_sts, shuffle=True, batch_size=batch_size)
train_loss_sts = losses.CosineSimilarityLoss(model=model)


logging.info("Read STSbenchmark dev dataset")
dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

# Configure the training
num_epochs = 4

warmup_steps = math.ceil(len(train_dataloader_sts) * num_epochs / batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Here we define the two train objectives: train_dataloader_nli with train_loss_nli (i.e., SoftmaxLoss for NLI data)
# and train_dataloader_sts with train_loss_sts (i.e., CosineSimilarityLoss for STSbenchmark data)
# You can pass as many (dataloader, loss) tuples as you like. They are iterated in a round-robin way.
train_objectives = [(train_dataloader_nli, train_loss_nli), (train_dataloader_sts, train_loss_sts)]

# Train the model
model.fit(train_objectives=train_objectives,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )



##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_data = SentencesDataset(examples=sts_reader.get_examples("sts-test.csv"), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluator(test_dataloader)

model.evaluate(evaluator)
