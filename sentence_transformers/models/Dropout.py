import torch
from torch import Tensor
from torch import nn
from typing import Dict
import os
import json


class Dropout(nn.Module):
    """Dropout layer.

    :param dropout: Sets a dropout value for dense layer.
    """
    def __init__(self, dropout: float = 0.2):
        super(Dropout, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, features: Dict[str, Tensor]):
        features.update({'sentence_embedding': self.dropout_layer(features['sentence_embedding'])})
        return features

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump({'dropout': self.dropout}, fOut)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        model = Dropout(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))
        return model
