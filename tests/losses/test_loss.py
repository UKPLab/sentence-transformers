from __future__ import annotations

import pytest
import torch

from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.losses import ContrastiveLoss

model = SentenceTransformer('all-MiniLM-L6-v2')


example_pairs = [
    InputExample(texts=['This is a positive example', 'This is a similar positive example'], label=1.0),
    InputExample(texts=['This is a positive example', 'This is a negative example'], label=0.0),
]


loss_test_cases = [
    (ContrastiveLoss, {"model": model}, example_pairs, 0.03),
]

@pytest.mark.parametrize("loss_class, loss_args, examples, expected_loss", loss_test_cases)
def test_loss_function(loss_class, loss_args, examples, expected_loss):
    loss_fn = loss_class(**loss_args)

    device = next(model.parameters()).device

    features_1 = model.tokenize([example.texts[0] for example in examples])
    features_2 = model.tokenize([example.texts[1] for example in examples])

    features_1 = {k: v.to(device) for k, v in features_1.items()}
    features_2 = {k: v.to(device) for k, v in features_2.items()}

    labels = torch.tensor([example.label for example in examples]).to(device)

    loss_value = loss_fn.forward([features_1, features_2], labels)

    assert isinstance(loss_value, torch.Tensor)
    assert loss_value.item() >= 0

    if expected_loss is not None:
        assert abs(loss_value.item() - expected_loss) <= 1e-1
