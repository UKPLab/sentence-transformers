
from sentence_transformers import SentenceTransformer, LoggingHandler
import logging
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.evaluation import TripletEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

#Important, you need to shield your code with if __name__. Otherwise, CUDA runs into issues when spawning new processes.
if __name__ == '__main__':
    #Set params
    data_stream_size = 50  #Size of the data that is loaded into memory at once
    chunk_size = 1024  #Size of the chunks that are sent to each process
    encode_batch_size = 128  #Batch size of the model
    
    #model_name = 'distilroberta-base'
    num_epochs = 1
    eval_dataset_path = './mmarco-3000-all.tsv.gz'
    batch_size_pairs = 10
    batch_size_triplets = 10
    max_seq_length = 256
    use_amp = True                  #Set to False, if you use a CPU or your GPU does not support FP16 operations
    evaluation_steps = 1000
    warmup_steps = 500


    #Load a large dataset in streaming mode. more info: https://huggingface.co/docs/datasets/stream
    
    dataset_name = "CloverSearch/cc-news-mutlilingual"
    dataset = load_dataset(dataset_name, streaming=True, split="train")
    train_dataloader = DataLoader(dataset.with_format("torch"), batch_size=data_stream_size)

    #Define the model

    #model_name = "nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large"
    #model_name = "./model_mMiniLMv2-L6-H384-distilled-from-XLMR-Large_ted2020_pairs"
    #model_name = "./model_mMiniLMv2-L12-H384-distilled-from-XLMR-Large_ted2020_pairs"
    model_name = "./model_model_mMiniLMv2-L6-H384-distilled-from-XLMR-Large_ted2020_pairs_globalvoices_pairs"
    #model = SentenceTransformer(model_name)
    

    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


    model_save_path = "./model_clover" + "_" + model_name.split("/")[1] + "_" + dataset_name.split("/")[1]
    #pool = model.start_multi_process_pool()
    train_loss = losses.MultipleNegativesRankingLoss(model)

    #lang_keys = ["english", "vietnamese", "french", "chinese", "german", "indonesian", "italian", "portuguese", "russian", "spanish", "arabic", "dutch", "hindi", "japanese"]

    # dev_samples = []
    # for lang in lang_keys:
    #     mmarco = load_dataset("unicamp-dl/mmarco", lang, streaming=True)
    #     mmarco_head = mmarco["train"].take(2_000)
    #     for row in mmarco_head:
    #         #score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
    #         dev_samples.append(InputExample(texts=[row['query'], row['positive'], row['negative']]))
    dev_samples = []
    with gzip.open(eval_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.reader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            dev_samples.append(InputExample(texts=[row[0], row[1], row[2]]))

    dev_evaluator = TripletEvaluator.from_input_examples(dev_samples, name='mmarco')
    print("evaluator created... ")

    model.fit(train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        evaluator=dev_evaluator,
        evaluation_steps=evaluation_steps,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
        use_amp=use_amp,
        checkpoint_path=model_save_path,
        checkpoint_save_steps = 1_000_000, #global voices is 2million, ted2020 is 10million pairs
        checkpoint_save_total_limit = 3,
        steps_per_epoch = 31_790_000  
        )
        
        
        
        #batch_emb = model.encode_multi_process(sentences, pool, chunk_size=chunk_size, batch_size=encode_batch_size)
    #print("Embeddings computed for 1 batch. Shape:", batch_emb.shape)

    #Optional: Stop the proccesses in the pool
    #model.stop_multi_process_pool(pool)