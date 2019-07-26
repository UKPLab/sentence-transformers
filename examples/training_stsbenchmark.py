"""
This is a simple training example. The system loads the pre-trained bert-base-nli-mean-tokens models from the server.
It then fine-tunes this model for some epochs on the STS benchmark dataset.
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import  SentenceTransformer, LossFunction, TrainConfig, SentencesDataset, LoggingHandler, EmbeddingSimilarityEvaluator, EmbeddingSimilarity
from sentence_transformers.dataset_readers import STSDataReader
import logging


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the dataset
train_batch_size = 32
sts_reader = STSDataReader('datasets/stsbenchmark', normalize_scores=True)


# Load a pre-trained sentence transformer model and set the loss function to COSINE_SIMILARITY
model = SentenceTransformer('bert-base-nli-mean-tokens')
model.transformer_model.sentence_transformer_config.loss_function = LossFunction.COSINE_SIMILARITY

# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark train dataset")
train_data = SentencesDataset(sts_reader.get_examples('sts-train.csv'), model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size, collate_fn=model.smart_batching_collate())

logging.info("Read STSbenchmark dev dataset")
dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size, collate_fn=model.smart_batching_collate())
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

# Configure the training. We skip evaluation in this example
num_epochs = 4
model_save_path = 'output/basic_training_stsbenchmark'
warmup_steps = math.ceil(len(train_data)*num_epochs/train_batch_size/10) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))
train_config = TrainConfig(learning_rate=2e-5,
                           weight_decay=0.01,
                           epochs=num_epochs,
                           output_path=model_save_path,
                           save_best_model=True,
                           evaluator=evaluator,
                           warmup_steps=warmup_steps)



# Train the model
model.train(dataloader=train_dataloader, train_config=train_config)




##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_data = SentencesDataset(examples=sts_reader.get_examples("sts-test.csv"), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=train_batch_size, collate_fn=model.smart_batching_collate())
evaluator = EmbeddingSimilarityEvaluator(test_dataloader, EmbeddingSimilarity.COSINE)

model.evaluate(evaluator)
