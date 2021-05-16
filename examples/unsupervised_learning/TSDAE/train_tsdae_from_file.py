"""
This file loads sentences from a provided text file. It is expected, that the there is one sentence per line in that text file.

TSDAE will be training using these sentences. Checkpoints are stored every 500 steps to the output folder.

Usage:
python train_tsdae_from_file.py path/to/sentences.txt

"""
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, datasets, losses
import logging
import gzip
from torch.utils.data import DataLoader
from datetime import datetime
import sys
import tqdm

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Train Parameters
model_name = 'bert-base-uncased'
batch_size = 8

#Input file path (a text file, each line a sentence)
if len(sys.argv) < 2:
    print("Run this script with: python {} path/to/sentences.txt".format(sys.argv[0]))
    exit()

filepath = sys.argv[1]

# Save path to store our model
output_name = ''
if len(sys.argv) >= 3:
    output_name = "-"+sys.argv[2].replace(" ", "_").replace("/", "_").replace("\\", "_")

model_output_path = 'output/train_tsdae{}-{}'.format(output_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


################# Read the train corpus  #################
train_sentences = []
with gzip.open(filepath, 'rt', encoding='utf8') if filepath.endswith('.gz') else open(filepath, encoding='utf8') as fIn:
    for line in tqdm.tqdm(fIn, desc='Read file'):
        line = line.strip()
        if len(line) >= 10:
            train_sentences.append(line)


logging.info("{} train sentences".format(len(train_sentences)))

################# Intialize an SBERT model #################

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


logging.info("Start training")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    weight_decay=0,
    scheduler='constantlr',
    optimizer_params={'lr': 3e-5},
    show_progress_bar=True,
    checkpoint_path=model_output_path,
    use_amp=False                #Set to True, if your GPU supports FP16 cores
)





