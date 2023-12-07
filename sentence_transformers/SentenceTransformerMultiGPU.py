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
        # Get the base model
        self.base_model = self.model._first_module().auto_model
        # Initialize the loss
        self.loss = loss(self.model)

    def training_step(self, batch, batch_idx):
        ''' The training function used by Pytorch Lightning '''
        features, labels = batch
        loss_value = self.loss(features, labels)

        self.log("train_loss", loss_value, batch_size=len(batch), sync_dist=True, prog_bar=True)
        return loss_value
    
    def configure_optimizers(self):
        ''' Optimizer configuration '''
        optimizer = optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=2e-8)
        return optimizer

