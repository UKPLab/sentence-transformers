import logging
import os
import csv
import numpy as np
from typing import List, Union
import math
from tqdm.autonotebook import trange

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from sentence_transformers import InputExample
from sentence_transformers.evaluation import SentenceEvaluator


logger = logging.getLogger(__name__)

class LossEvaluator(SentenceEvaluator):

    def __init__(self, loader, loss_model: nn.Module = None, name: str = '', show_progress_bar: bool = False, write_csv: bool = True):
        
        """
        Evaluate a model based on the loss function.
        The returned score is loss value.
        The results are written in a CSV and Tensorboard logs.
        :param loader: Data loader object
        :param loss_model: loss module object
        :param name: Name for the output
        :param show_progress_bar: If true, prints a progress bar
        :param write_csv: Write results to a CSV file
        """
        
        self.loader = loader
        self.write_csv = write_csv
        self.name = name
        self.loss_model = loss_model
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "loss_evaluation"+("_"+name if name else '')+"_results.csv"
        self.csv_headers = ["epoch", "steps", "loss"]
        
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"
        
        model.eval()
        
        loss_value = 0
        num_batches = len(self.loader)
        data_iterator = iter(self.loader)
        with torch.no_grad():
            for _ in trange(num_batches, desc="Iteration", smoothing=0.05):
                sentence_features, labels = next(data_iterator)
                loss_value += self.loss_model(sentence_features, labels).item()
        
        final_loss = loss_value/num_batches
        if output_path is not None and self.write_csv:

            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            tensor_writer = SummaryWriter(os.path.join(output_path,'runs'))
            
            with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, final_loss])
                
            # ...log the running loss
            tensor_writer.add_scalar('validation loss',
                            final_loss,
                            epoch)
        
        model.train()
        self.loss_model.zero_grad()
        self.loss_model.train()
        
        return final_loss