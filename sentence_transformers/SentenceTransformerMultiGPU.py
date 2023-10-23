import torch
from torch import optim
import lightning.pytorch as pl
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer, models

class SentenceTransformerMultiGPU(pl.LightningModule):
    ''' Train a SentenceTransformer model using Pytorch Lightning '''
    
    def __init__(self, model, loss, model_device):
        super().__init__()
        self.model_device = model_device
        # Load the transformer model you want to train as a sentence transformer
        embedding_model = models.Transformer(model)
        # Add the pooling layer
        pooling = models.Pooling(embedding_model.get_word_embedding_dimension())
        # Construct the sentence transformer
        self.model = SentenceTransformer(modules=[embedding_model, pooling])
        # Freeze the all the model except the dense layer
        self.base_model = self.model._first_module().auto_model
        # for name, param in self.base_model.named_parameters():
        #     if name == 'pooler.dense.weight' or name == 'pooler.dense.bias':
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        # Define the loss
        self.loss = loss(self.model)

    def training_step(self, batch, batch_idx):
        ''' The training function used by Pytorch Lightning '''
        features, labels = batch
        # loss_value = self.loss(features, labels)
        # sentence_list_1 = batch[0]
        # sentence_list_2 = [example[0] for example in batch]
        # score_list = [example[2] for example in batch]

        # sentence1_embeddings = self.model.encode(sentence_list_1, convert_to_tensor=True, device=self.model_device)
        # sentence2_embeddings = self.model.encode(sentence_list_2, convert_to_tensor=True, device=self.model_device)
        # loss_value = self.loss([{'sentence_embedding': sentence1_embeddings}, {'sentence_embedding': sentence2_embeddings}], score_list)
        # loss_value = self.loss([sentence_list_1, sentence_list_2], score_list)
        loss_value = self.loss(features, labels)

        self.log("train_loss", loss_value, batch_size=len(batch), sync_dist=True, prog_bar=True)
        return loss_value
    
    # def validation_step(self, batch, batch_idx):
    #     '''
    #     The training function used by Pytorch Lightning, 
    #     it uses cosine_pearson and cosine_spearman metrics for the evaluation of the model
    #     '''
    #     features, labels = batch
    #     print(features)
    #     evaluator = EmbeddingSimilarityEvaluator(features[0], features[1], labels, batch_size=len(batch))
    #     metrics = evaluator(self.model)
    #     print(metrics)
    #     cosine_metrics = {
    #         "cos_pearson": metrics["cosine_pearson"],
    #         "cos_spearman": metrics["cosine_spearman"]
    #     }
    #     self.log_dict(cosine_metrics, batch_size=len(batch), sync_dist=True, prog_bar=True, on_epoch=True)

    #     # evaluator = EmbeddingSimilarityEvaluator.from_input_examples(batch, name='sts-benchmark-val')
    #     # evaluator(self.model)

    def configure_optimizers(self):
        ''' Optimizer configuration '''
        optimizer = optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=2e-8)
        return optimizer

