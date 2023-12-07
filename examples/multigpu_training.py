import lightning.pytorch as pl
from sentence_transformers import losses
from sentence_transformers import InputExample
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformerMultiGPU

st_model = SentenceTransformerMultiGPU("prajjwal1/bert-small", losses.CosineSimilarityLoss, "cpu")

class DatasetClass(Dataset):
    def __init__(self):
        self.text1_list = ["hi", "bye", "I love you"]
        self.text2_list = ["hello", "bye bye", "I hate you"]
        self.score_list = [0.7, 0.9, 0.0]

    def __len__(self):
        return len(self.text1_list)

    def __getitem__(self, idx):
        text1 = self.text1_list[idx]
        text2 = self.text2_list[idx]
        score = self.score_list[idx]
        return InputExample(texts=[text1, text2], label=score)

def collate(batch):
    return batch

train_dataset = DatasetClass()
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=st_model.model.smart_batching_collate)
val_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=st_model.model.smart_batching_collate)

trainer = pl.Trainer(max_epochs = 4,
                     accelerator = 'cpu',
                     devices = 1,
                     log_every_n_steps = 50)
trainer.fit(st_model, train_dataloader, val_dataloader)