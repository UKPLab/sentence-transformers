import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import GISTEmbedLoss

def test_gist_embed_loss():
    model = SentenceTransformer("all-MiniLM-L6-v2").to("cpu")
    guide = SentenceTransformer("all-MiniLM-L6-v2").to("cpu")

    loss_function = GISTEmbedLoss(model, guide)

    batch = [
        {"input_ids": torch.randint(0, 1000, (3, 10)), "attention_mask": torch.ones((3, 10))},
        {"input_ids": torch.randint(0, 1000, (3, 10)), "attention_mask": torch.ones((3, 10))},
        {"input_ids": torch.randint(0, 1000, (3, 10)), "attention_mask": torch.ones((3, 10))},
    ]
    labels = torch.zeros(3)

    loss = loss_function(batch, labels)
    
    print(loss.item())
    
    assert loss is not None, "Loss should not be None"
    assert loss.item() > 0, "Loss should be greater than 0"

