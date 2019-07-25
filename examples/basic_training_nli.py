"""
This is a simple training example. The system trains on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset
"""
from torch.utils.data import DataLoader
import math
import sentence_transformers.models
from sentence_transformers import SentenceTransformerConfig, SentenceTransformer, LossFunction, TrainConfig, SentencesDataset, LoggingHandler, EmbeddingSimilarityEvaluator, EmbeddingSimilarity
from sentence_transformers.dataset_readers import NLIDataReader, STSDataReader
import logging


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the dataset
train_batch_size = 32
nli_reader = NLIDataReader('datasets/AllNLI')
sts_reader = STSDataReader('datasets/stsbenchmark')
train_num_labels = nli_reader.get_num_labels()

# Create a Sentence BERT model with Softmax loss function
sentence_transformer_config = SentenceTransformerConfig(
                                            model=sentence_transformers.models.BERT,
                                            tokenizer_model='bert-base-uncased',
                                            do_lower_case=True,
                                            max_seq_length=64,
                                            pooling_mode_cls_token=False,
                                            pooling_mode_max_tokens=False,
                                            pooling_mode_mean_tokens=True,
                                            loss_function=LossFunction.SOFTMAX,
                                            softmax_num_labels=train_num_labels,
                                            softmax_concatenation_sent_rep=True,
                                            softmax_concatenation_sent_difference=True,
                                            softmax_concatenation_sent_multiplication=False)

model = SentenceTransformer(sentence_transformer_config=sentence_transformer_config)

# Convert the dataset to a DataLoader ready for training
logging.info("Read AllNLI train dataset")
train_data = SentencesDataset(nli_reader.get_examples('train.gz'), model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size, collate_fn=model.smart_batching_collate())


logging.info("Read STSbenchmark dev dataset")
dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size, collate_fn=model.smart_batching_collate())
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

# Configure the training. We skip evaluation in this example
num_epochs = 1
model_save_path = 'output/basic_training_nli'
warmup_steps = math.ceil(len(train_data)*num_epochs/train_batch_size/10) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))
train_config = TrainConfig(epochs=num_epochs,
                           evaluation_steps=1000,
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
