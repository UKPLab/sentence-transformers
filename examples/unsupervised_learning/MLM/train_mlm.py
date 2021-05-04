"""
This file runs Masked Language Model. You provide a training file. Each line is interpreted as a sentence / paragraph.
Optionally, you can also provide a dev file.

The fine-tuned model is stored in the output/model_name folder.

Usage:
python train_mlm.py model_name data/train_sentences.txt [data/dev_sentences.txt]
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments
import sys
import gzip
from datetime import datetime

if len(sys.argv) < 3:
    print("Usage: python train_mlm.py model_name data/train_sentences.txt [data/dev_sentences.txt]")
    exit()

model_name = sys.argv[1]
per_device_train_batch_size = 64

save_steps = 1000               #Save model every 1k steps
num_train_epochs = 3            #Number of epochs
use_fp16 = False                #Set to True, if your GPU supports FP16 operations
max_length = 100                #Max length for a text input
do_whole_word_mask = True       #If set to true, whole words are masked
mlm_prob = 15                   #Probability that a word is replaced by a [MASK] token

# Load the model
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


output_dir = "output/{}-{}".format(model_name.replace("/", "_"),  datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
print("Save checkpoints to:", output_dir)


##### Load our training datasets

train_sentences = []
train_path = sys.argv[2]
with gzip.open(train_path, 'rt', encoding='utf8') if train_path.endswith('.gz') else  open(train_path, 'r', encoding='utf8') as fIn:
    for line in fIn:
        line = line.strip()
        if len(line) >= 10:
            train_sentences.append(line)

print("Train sentences:", len(train_sentences))

dev_sentences = []
if len(sys.argv) >= 4:
    dev_path = sys.argv[3]
    with gzip.open(dev_path, 'rt', encoding='utf8') if dev_path.endswith('.gz') else open(dev_path, 'r', encoding='utf8') as fIn:
        for line in fIn:
            line = line.strip()
            if len(line) >= 10:
                dev_sentences.append(line)

print("Dev sentences:", len(dev_sentences))

#A dataset wrapper, that tokenizes our data on-the-fly
class TokenizedSentencesDataset:
    def __init__(self, sentences, tokenizer, max_length, cache_tokenization=False):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization

    def __getitem__(self, item):
        if not self.cache_tokenization:
            return self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)

        if isinstance(self.sentences[item], str):
            self.sentences[item] = self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)

train_dataset = TokenizedSentencesDataset(train_sentences, tokenizer, max_length)
dev_dataset = TokenizedSentencesDataset(dev_sentences, tokenizer, max_length, cache_tokenization=True) if len(dev_sentences) > 0 else None


##### Training arguments

if do_whole_word_mask:
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    evaluation_strategy="steps",
    per_device_train_batch_size=per_device_train_batch_size,
    eval_steps=save_steps,
    save_steps=save_steps,
    logging_steps=save_steps,
    save_total_limit=1,
    prediction_loss_only=True,
    fp16=use_fp16
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset
)

trainer.train()

print("Save model to:", output_dir)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Training done")