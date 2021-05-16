"""
This file loads sentences from a provided text file. It is expected, that the there is one sentence per line in that text file.

CT will be training using these sentences. Checkpoints are stored every 500 steps to the output folder.

Usage:
python train_ct_from_file.py path/to/sentences.txt

"""
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer
import logging
from datetime import datetime
import gzip
import sys
import tqdm
from torch.utils.data import DataLoader

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

## Training parameters
model_name = 'distilbert-base-uncased'
batch_size = 128
num_epochs = 1
max_seq_length = 75

#Input file path (a text file, each line a sentence)
if len(sys.argv) < 2:
    print("Run this script with: python {} path/to/sentences.txt".format(sys.argv[0]))
    exit()

filepath = sys.argv[1]

# Save path to store our model
output_name = ''
if len(sys.argv) >= 3:
    output_name = "-"+sys.argv[2].replace(" ", "_").replace("/", "_").replace("\\", "_")

model_output_path = 'output/train_ct-improved{}-{}'.format(output_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

################# Read the train corpus  #################
train_sentences = []
with gzip.open(filepath, 'rt', encoding='utf8') if filepath.endswith('.gz') else open(filepath, encoding='utf8') as fIn:
    for line in tqdm.tqdm(fIn, desc='Read file'):
        line = line.strip()
        if len(line) >= 10:
            train_sentences.append(line)


logging.info("Train sentences: {}".format(len(train_sentences)))

# A regular torch DataLoader and as loss we use losses.ContrastiveTensionLossInBatchNegatives
train_dataloader = DataLoader(train_sentences, batch_size=batch_size, shuffle=True, drop_last=True)
train_loss = losses.ContrastiveTensionLossInBatchNegatives(model)


warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          optimizer_params={'lr': 5e-5},
          checkpoint_path=model_output_path,
          show_progress_bar=True,
          use_amp=False  # Set to True, if your GPU supports FP16 cores
          )
