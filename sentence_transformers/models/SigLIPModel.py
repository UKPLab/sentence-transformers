from typing import Union
from torch import nn
import transformers
import torch
from PIL import Image


class SigLIPModel(nn.Module):
    def __init__(self, model_name: str = "google/siglip-so400m-patch14-384", processor_name=None):
        super(SigLIPModel, self).__init__()

        if processor_name is None:
            processor_name = model_name

        self.model = transformers.AutoModel.from_pretrained(model_name)
        self.processor = transformers.AutoProcessor.from_pretrained(processor_name)

    def __repr__(self):
        return "SigLIPModel()"

    def forward(self, features):
        image_embeds = []
        text_embeds = []

        if "pixel_values" in features:
            image_embeds = self.model.get_image_features(features["pixel_values"])

        if "input_ids" in features:
            text_embeds = self.model.get_text_features(
                input_ids=features.get("input_ids"),
                attention_mask=features.get("attention_mask", None),
                position_ids=features.get("position_ids", None),
                output_attentions=features.get("output_attentions", None),
                output_hidden_states=features.get("output_hidden_states", None),
            )

        sentence_embedding = []
        image_features = iter(image_embeds)
        text_features = iter(text_embeds)

        for idx, input_type in enumerate(features["image_text_info"]):
            if input_type == 0:
                sentence_embedding.append(next(image_features))
            else:
                sentence_embedding.append(next(text_features))

        features["sentence_embedding"] = torch.stack(sentence_embedding).float()

        return features

    def tokenize(self, texts, padding: Union[str, bool] = "max_length"):
        images = []
        texts_values = []
        image_text_info = []

        for idx, data in enumerate(texts):
            if isinstance(data, Image.Image):  # An Image
                images.append(data)
                image_text_info.append(0)
            else:  # A text
                texts_values.append(data)
                image_text_info.append(1)

        encoding = {}
        if len(texts_values):
            encoding = self.processor.tokenizer(texts_values, return_tensors="pt", padding=padding)

        if len(images):
            image_features = self.processor.image_processor(images, return_tensors="pt")
            encoding["pixel_values"] = image_features.pixel_values

        encoding["image_text_info"] = image_text_info
        return encoding

    def save(self, output_path: str):
        self.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)

    @staticmethod
    def load(input_path: str):
        return SigLIPModel(model_name=input_path)
