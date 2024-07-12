"""
This file loads sentences from a provided text file. It is expected, that the there is one sentence per line in that text file.

CT will be training using these sentences. Checkpoints are stored every 500 steps to the output folder.

Usage:
python train_ct_from_file.py path/to/sentences.txt

"""

import gzip
import logging
import math
import sys
from datetime import datetime

import tqdm

from sentence_transformers import LoggingHandler, SentenceTransformer, losses, models

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout

## Training parameters
model_name = "distilbert-base-uncased"
batch_size = 16
pos_neg_ratio = 8  # batch_size must be devisible by pos_neg_ratio
num_epochs = 1
max_seq_length = 75

# Input file path (a text file, each line a sentence)
if len(sys.argv) < 2:
    print(f"Run this script with: python {sys.argv[0]} path/to/sentences.txt")
    exit()

filepath = sys.argv[1]

# Save path to store our model
output_name = ""
if len(sys.argv) >= 3:
    output_name = "-" + sys.argv[2].replace(" ", "_").replace("/", "_").replace("\\", "_")

model_output_path = "output/train_ct{}-{}".format(output_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

################# Read the train corpus  #################
train_sentences = []
with gzip.open(filepath, "rt", encoding="utf8") if filepath.endswith(".gz") else open(
    filepath, encoding="utf8"
) as fIn:
    for line in tqdm.tqdm(fIn, desc="Read file"):
        line = line.strip()
        if len(line) >= 10:
            train_sentences.append(line)


logging.info(f"Train sentences: {len(train_sentences)}")

# For ContrastiveTension we need a special data loader to construct batches with the desired properties
train_dataloader = losses.ContrastiveTensionDataLoader(
    train_sentences, batch_size=batch_size, pos_neg_ratio=pos_neg_ratio
)

# As loss, we losses.ContrastiveTensionLoss
train_loss = losses.ContrastiveTensionLoss(model)


warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info(f"Warmup-steps: {warmup_steps}")

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    optimizer_params={"lr": 5e-5},
    checkpoint_path=model_output_path,
    show_progress_bar=True,
    use_amp=False,  # Set to True, if your GPU supports FP16 cores
)
