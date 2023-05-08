
from sentence_transformers import SentenceTransformer, LoggingHandler, InputExample
from sentence_transformers import models, util, datasets, evaluation, losses
import logging
import os
import gzip
from torch.utils.data import DataLoader
from datetime import datetime


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Some training parameters. For the example, we use a batch_size of 128, a max sentence length (max_seq_length)
# of 32 word pieces and as model roberta-base
model_name = 'roberta-base'
batch_size = 128
max_seq_length = 32
num_epochs = 1

################# Download AskUbuntu and extract training corpus  #################
askubuntu_folder = 'data/askubuntu'
output_path = 'output/askubuntu-simcse-{}-{}-{}'.format(model_name, batch_size, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

## Download the AskUbuntu dataset from https://github.com/taolei87/askubuntu
for filename in ['text_tokenized.txt.gz', 'dev.txt', 'test.txt', 'train_random.txt']:
    filepath = os.path.join(askubuntu_folder, filename)
    if not os.path.exists(filepath):
        util.http_get('https://github.com/taolei87/askubuntu/raw/master/'+filename, filepath)

# Read the corpus
corpus = {}
dev_test_ids = set()
with gzip.open(os.path.join(askubuntu_folder, 'text_tokenized.txt.gz'), 'rt', encoding='utf8') as fIn:
    for line in fIn:
        splits = line.strip().split("\t")
        id = splits[0]
        title = splits[1]
        corpus[id] = title

# Read dev & test dataset
def read_eval_dataset(filepath):
    dataset = []
    with open(filepath) as fIn:
        for line in fIn:
            query_id, relevant_id, candidate_ids, bm25_scores = line.strip().split("\t")
            if len(relevant_id) == 0:   #Skip examples without relevant entries
                continue

            relevant_id = relevant_id.split(" ")
            candidate_ids = candidate_ids.split(" ")
            negative_ids = set(candidate_ids) - set(relevant_id)
            dataset.append({
                'query': corpus[query_id],
                'positive': [corpus[pid] for pid in relevant_id],
                'negative': [corpus[pid] for pid in negative_ids]
            })
            dev_test_ids.add(query_id)
            dev_test_ids.update(candidate_ids)
    return dataset

dev_dataset = read_eval_dataset(os.path.join(askubuntu_folder, 'dev.txt'))
test_dataset = read_eval_dataset(os.path.join(askubuntu_folder, 'test.txt'))


## Now we need a list of train sentences.
## In this example we simply use all sentences that don't appear in the train/dev set
train_sentences = []
for id, sentence in corpus.items():
    if id not in dev_test_ids:
        train_sentences.append(InputExample(texts=[sentence, sentence]))

logging.info("{} train sentences".format(len(train_sentences)))

################# Intialize an SBERT model #################


word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

# Apply mean pooling
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

################# Train the model #################

# As Loss function, we use MultipleNegativesRankingLoss
train_dataloader = DataLoader(train_sentences, batch_size=batch_size, shuffle=True, drop_last=True)
train_loss = losses.MultipleNegativesRankingLoss(model)

# Create a dev evaluator
dev_evaluator = evaluation.RerankingEvaluator(dev_dataset, name='AskUbuntu dev')
test_evaluator = evaluation.RerankingEvaluator(test_dataset, name='AskUbuntu test')

logging.info("Dev performance before training")
dev_evaluator(model)

warmup_steps = int(num_epochs*len(train_dataloader)*0.1)

logging.info("Start training")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    evaluation_steps=100,
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=output_path,
    show_progress_bar=True,
    use_amp=True                #If your GPU does not have FP16 cores, set use_amp=False
)

latest_output_path = output_path + "-latest"
model.save(latest_output_path)

### Run test evaluation on the latest model. This is equivalent to not having a dev dataset
model = SentenceTransformer(latest_output_path)
test_evaluator(model)


