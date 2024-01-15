import torch
from torch import Tensor
from torch import nn
from typing import Dict
import os
import json
from ..util import fullname, import_from_string


class Dense(nn.Module):
    """Feed-forward function with  activiation function.

    This layer takes a fixed-sized sentence embedding and passes it through a feed-forward layer. Can be used to generate deep averaging networks (DAN).

    :param in_features: Size of the input dimension
    :param out_features: Output size
    :param bias: Add a bias vector
    :param activation_function: Pytorch activation function applied on output
    :param init_weight: Initial value for the matrix of the linear layer
    :param init_bias: Initial value for the bias of the linear layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation_function=nn.Tanh(),
        init_weight: Tensor = None,
        init_bias: Tensor = None,
    ):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation_function = activation_function
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if init_weight is not None:
            self.linear.weight = nn.Parameter(init_weight)

        if init_bias is not None:
            self.linear.bias = nn.Parameter(init_bias)

    def forward(self, features: Dict[str, Tensor]):
        features.update({"sentence_embedding": self.activation_function(self.linear(features["sentence_embedding"]))})
        return features

    def get_sentence_embedding_dimension(self) -> int:
        return self.out_features

    def get_config_dict(self):
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "activation_function": fullname(self.activation_function),
        }

    def save(self, output_path):
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut)

        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    def __repr__(self):
        return "Dense({})".format(self.get_config_dict())

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        config["activation_function"] = import_from_string(config["activation_function"])()
        model = Dense(**config)
        model.load_state_dict(
            torch.load(os.path.join(input_path, "pytorch_model.bin"), map_location=torch.device("cpu"))
        )
        return model
