from typing import List
import pytest
from sentence_transformers import SentenceTransformer, InputExample, losses
import tqdm
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from transformers import set_seed
import torch
from torch.optim import Adam


@pytest.mark.parametrize(
    ["train_samples_mnrl", "train_samples_cmnrl", "same_grad"],
    [
        (
            [
                InputExample(texts=[q, p, n])
                for q, p, n in zip(
                    ["aaa", "bbb", "ccc", "ddd", "eee"],
                    ["aas", "bbs", "ccs", "dds", "ees"],
                    ["xxx", "yyy", "zzz", "kkk", "fff"],
                )
            ],
            [
                InputExample(texts=[q, p, n])
                for q, p, n in zip(
                    ["aaa", "bbb", "ccc", "ddd", "eee"],
                    ["aas", "bbs", "ccs", "dds", "ees"],
                    ["xxx", "yyy", "zzz", "kkk", "fff"],
                )
            ],
            True,
        ),
        (
            [
                InputExample(texts=[q, p, n])
                for q, p, n in zip(
                    ["adsa", "czx", "dsada"],
                    ["b", "fas", "xcz"],
                    ["c", "yyy", "asdas"],
                )
            ],
            [
                InputExample(texts=[q, p, n])
                for q, p, n in zip(
                    ["aaa", "bbb", "ccc", "ddd", "eee"],
                    ["aas", "bbs", "ccs", "dds", "ees"],
                    ["xxx", "yyy", "zzz", "kkk", "fff"],
                )
            ],
            False,
        ),
    ],
)
def test_cmnrl_same_grad(
    train_samples_mnrl: List[InputExample],
    train_samples_cmnrl: List[InputExample],
    same_grad: bool,
):
    # Given:
    sbert = SentenceTransformer("distilbert-base-uncased")
    sbert.to("cpu")
    optimizer = Adam(sbert.parameters())
    train_samples_mnrl
    train_samples_cmnrl
    same_grad

    # When:
    # First run with MNRL
    set_seed(42)
    optimizer.zero_grad()
    loss_mnrl = losses.MultipleNegativesRankingLoss(sbert)
    loss_mnrl.eval()  # To distable dropout
    loss_mnrl_value: torch.Tensor = loss_mnrl.forward(
        *sbert.smart_batching_collate(train_samples_mnrl)
    )
    loss_mnrl_value.backward()
    grad_expected = {
        name: p.grad.clone()
        for name, p in loss_mnrl.named_parameters()
        if p.grad is not None
    }

    # Then run with this cached version:
    set_seed(42)
    optimizer.zero_grad()
    loss_cmnrl = CachedMultipleNegativesRankingLoss(sbert, mini_batch_size=2)
    loss_cmnrl.eval()  # To distable dropout
    loss_cmnrl_value = loss_cmnrl.forward(
        *sbert.smart_batching_collate(train_samples_cmnrl)
    )
    loss_cmnrl_value.backward()
    grad = {
        name: p.grad.clone()
        for name, p in loss_cmnrl.named_parameters()
        if p.grad is not None
    }

    # Then:
    if same_grad:
        assert pytest.approx(loss_mnrl_value.item()) == loss_cmnrl_value.item()
    else:
        assert pytest.approx(loss_mnrl_value.item()) != loss_cmnrl_value.item()

    nclose = 0
    for name in tqdm.tqdm(grad_expected):
        nclose += torch.allclose(grad[name], grad_expected[name], 1e-6, 1e-6)

    if same_grad:
        assert nclose == len(grad_expected)
    else:
        assert nclose != len(grad_expected)
