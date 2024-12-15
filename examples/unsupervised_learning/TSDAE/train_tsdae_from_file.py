import numpy as np
from datasets import load_dataset
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

from sentence_transformers import (
SentenceTransformer,
SentenceTransformerTrainer,
SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import DenoisingAutoEncoderLoss

# 1. Load a model to finetune with 2. (Optional) model card data
model = SentenceTransformer(
    "bert-base-cased",
)

# 3. Load a dataset to finetune on
dataset = load_dataset("sentence-transformers/all-nli", "triplet")
train_dataset = dataset["train"].select_columns(["anchor"]).select(range(100_000))
eval_dataset = dataset["dev"].select_columns(["anchor"])
test_dataset = dataset["test"].select_columns(["anchor"])
# Now we have 3 datasets, each with one column of text (called "anchor", but the name doesn't matter)
# Now we need to convert the dataset into 2 columns: (damaged_sentence, original_sentence), see https://sbert.net/docs/sentence_transformer/loss_overview.html

def noise_fn(text, del_ratio=0.6):
    words = word_tokenize(text)
    n = len(words)
    if n == 0:
        return text

    keep_or_not = np.random.rand(n) > del_ratio
    if sum(keep_or_not) == 0:
        keep_or_not[np.random.choice(n)] = True  # guarantee that at least one word remains
    words_processed = TreebankWordDetokenizer().detokenize(np.array(words)[keep_or_not])
    return {
        "damaged": words_processed,
        "original": text,
    }

train_dataset = train_dataset.map(noise_fn, input_columns="anchor", remove_columns="anchor")
eval_dataset = eval_dataset.map(noise_fn, input_columns="anchor", remove_columns="anchor")
test_dataset = test_dataset.map(noise_fn, input_columns="anchor", remove_columns="anchor")
# Now we have datasets with 2 columns, damaged & original (in that order). The "anchor" column is removed

# 4. Define a loss function
loss = DenoisingAutoEncoderLoss(model, decoder_name_or_path="bert-base-cased", tie_encoder_decoder=True)

# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/bert-base-cased-nli-tsdae",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="bert-base-cased-nli-tsdae",  # Will be used in W&B if `wandb` is installed
)

# 6. (Optional) Make an evaluator to evaluate before, during, and after training

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
)
trainer.train()

# 8. Save the trained model
model.save_pretrained("models/bert-base-cased-nli-tsdae/final")

# 9. (Optional) Push it to the Hugging Face Hub
model.push_to_hub("bert-base-cased-nli-tsdae")
