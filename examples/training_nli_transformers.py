"""
The system trains BERT on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset
"""
import argparse
from transformers.modeling_auto import MODEL_MAPPING
from transformers import ALL_PRETRAINED_MODEL_ARCHIVE_MAP
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
import logging
from datetime import datetime

MODEL_CLASSES = tuple(m.model_type for m in MODEL_MAPPING)
ALL_MODELS = tuple(ALL_PRETRAINED_MODEL_ARCHIVE_MAP)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="bert-base-uncased",
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument('--model_type', default="bert",
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES))
    parser.add_argument('--nli_dataset_path', default="datasets/AllNLI")
    parser.add_argument('--sts_dataset_path', default="datasets/stsbenchmark")
    parser.add_argument('--model_output_dir', default="output/")
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--evaluation_steps', default=1000, type=int)
    parser.add_argument("--fp16", action="store_true",
                        help="Use Apex Mixed Precision")

    args = parser.parse_args()

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    nli_reader = NLIDataReader(args.nli_dataset_path)
    sts_reader = STSDataReader(args.sts_dataset_path)
    train_num_labels = nli_reader.get_num_labels()
    model_save_path = args.model_output_dir + '/training_nli_' + args.model_name_or_path + '-' + \
                      datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Use BERT for mapping tokens to embeddings
    word_embedding_model = models.Transformers(model_name_or_path=args.model_name_or_path,
                                               model_type=args.model_type)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Convert the dataset to a DataLoader ready for training
    logging.info("Read AllNLI train dataset")
    train_data = SentencesDataset(nli_reader.get_examples('train.gz'), model=model)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                    num_labels=train_num_labels)

    logging.info("Read STSbenchmark dev dataset")
    dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=args.batch_size)
    evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

    warmup_steps = math.ceil(
        len(train_dataloader) * args.num_epochs / args.batch_size * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=args.num_epochs,
              evaluation_steps=args.evaluation_steps,
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              fp16=args.fp16
              )

    ##############################################################################
    #
    # Load the stored model and evaluate its performance on STS benchmark dataset
    #
    ##############################################################################

    model = SentenceTransformer(model_save_path)
    test_data = SentencesDataset(examples=sts_reader.get_examples("sts-test.csv"), model=model)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size)
    evaluator = EmbeddingSimilarityEvaluator(test_dataloader)

    model.evaluate(evaluator)
