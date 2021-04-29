
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
import logging
import os
import gzip
from torch.utils.data import DataLoader
from datetime import datetime
import sys

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

################# Download AskUbuntu and extract training corpus  #################
askubuntu_folder = 'data/askubuntu'
result_folder = 'output/askubuntu-tsdae-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
batch_size = 8

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
        train_sentences.append(sentence)


logging.info("{} train sentences".format(len(train_sentences)))

################# Intialize an SBERT model #################
model_name = sys.argv[1] if len(sys.argv) >= 2 else 'bert-base-uncased'
word_embedding_model = models.Transformer(model_name)
# Apply **cls** pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=False,
                               pooling_mode_cls_token=True,
                               pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

################# Train and evaluate the model (it needs about 1 hour for one epoch of AskUbuntu) #################
# We wrap our training sentences in the DenoisingAutoEncoderDataset to add deletion noise on the fly
train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=True)

# Create a dev evaluator
dev_evaluator = evaluation.RerankingEvaluator(dev_dataset, name='AskUbuntu dev')

logging.info("Dev performance before training")
dev_evaluator(model)

total_steps = 20000
logging.info("Start training")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    evaluation_steps=1000,
    epochs=1,
    steps_per_epoch=total_steps,
    weight_decay=0,
    scheduler='constantlr',
    optimizer_params={'lr': 3e-5},
    output_path=result_folder,
    show_progress_bar=True
)



