#!/bin/bash

export target_languages=("de","es","it","fr","ar","tr")
export source_languages=("en")

python examples/training/multilingual/make_multilingual.py \
    --train_batch_size="64" \
    --inference_batch_size="64" \
    --max_sentences_per_language="500000" \
    --train_max_sentence_length="250" \
    --num_epochs="5" \
    --num_warmup_steps="10000" \
    --num_evaluation_steps="1000" \
    --dev_sentences="1000" \
    --teacher_model_name="paraphrase-distilroberta-base-v2" \
    --student_model_name="xlm-roberta-base" \
    --max_seq_length="128" \
    --source_languages=$source_languages \
    --target_languages=$target_languages \
    --train_corpus="datasets/ted2020.tsv.gz" \
    --val_corpus="datasets/STS2017-extended.zip"
