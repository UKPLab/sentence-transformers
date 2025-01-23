import gzip
import logging
import random
import sys
import traceback
from datetime import datetime

import tqdm
from datasets import Dataset, load_dataset

from sentence_transformers import SentenceTransformer, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# Training parameters
model_name = "bert-base-uncased"
train_batch_size = 8
num_epochs = 1
max_seq_length = 75

output_dir = f"output/training_tsdae-{model_name.replace('/', '-')}-{train_batch_size}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# 1. Defining our sentence transformer model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), "cls")
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
# or to load a pre-trained SentenceTransformer model OR use mean pooling
# model = SentenceTransformer(model_name)
# model.max_seq_length = max_seq_length

# 2. We use a file provided by the user to train our model
# Input file path (a text file, each line a sentence)
if len(sys.argv) < 2:
    print(f"Run this script with: python {sys.argv[0]} path/to/sentences.txt")
    exit()

filepath = sys.argv[1]
train_sentences = []
with (
    gzip.open(filepath, "rt", encoding="utf8") if filepath.endswith(".gz") else open(filepath, encoding="utf8") as fIn
):
    for line in tqdm.tqdm(fIn, desc="Reading file"):
        line = line.strip()
        if len(line) >= 10:
            train_sentences.append(line)
dataset = Dataset.from_dict({"text": train_sentences})


def noise_fn(text, del_ratio=0.6):
    from nltk import word_tokenize
    from nltk.tokenize.treebank import TreebankWordDetokenizer

    words = word_tokenize(text)
    n = len(words)
    if n == 0:
        return text

    kept_words = [word for word in words if random.random() < del_ratio]
    # Guarantee that at least one word remains
    if len(kept_words) == 0:
        return {"noisy": random.choice(words)}

    noisy_text = TreebankWordDetokenizer().detokenize(kept_words)
    return {"noisy": noisy_text}


# TSDAE requires a dataset with 2 columns: a text column and a noisified text column
# Here we are using a function to delete some words, but you can use any other method to noisify your text
dataset = dataset.map(noise_fn, input_columns="text")
dataset = dataset.train_test_split(test_size=10000)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]
print(train_dataset)
print(train_dataset[0])
"""
Dataset({
    features: ['text', 'noisy'],
    num_rows: 990000
})
{
    'text': 'Oseltamivir is considered to be the primary antiviral drug used to combat avian influenza, commonly known as the bird flu.',
    'noisy': 'to be the primary antiviral drug used combat influenza commonly as the bird flu.',
}
"""

# 3. Define our training loss: https://sbert.net/docs/package_reference/sentence_transformer/losses.html#denoisingautoencoderLoss
# Note that this will likely result in warnings as we're loading 'model_name' as a decoder, but it likely won't
# have weights for that yet. This is fine, as we'll be training it from scratch.
train_loss = DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=True)

# 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=stsb_eval_dataset["sentence1"],
    sentences2=stsb_eval_dataset["sentence2"],
    scores=stsb_eval_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
)
logging.info("Evaluation before training:")
dev_evaluator(model)

# 5. Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    learning_rate=3e-5,
    num_train_epochs=1,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
    run_name="tsdae",  # Will be used in W&B if `wandb` is installed
)

# 6. Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
)
trainer.train()

# 7. Evaluate the model performance on the STS Benchmark test dataset
test_dataset = load_dataset("sentence-transformers/stsb", split="test")
test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=test_dataset["sentence1"],
    sentences2=test_dataset["sentence2"],
    scores=test_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-test",
)
test_evaluator(model)

# 8. Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)

# 9. (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-tsdae")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-tsdae')`."
    )
