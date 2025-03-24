from __future__ import annotations

import torch
import transformers
from PIL import Image
from torch import nn


class CLIPModel(nn.Module):
    save_in_root: bool = True

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", processor_name=None) -> None:
        super().__init__()

        if processor_name is None:
            processor_name = model_name

        self.model = transformers.CLIPModel.from_pretrained(model_name)
        self.processor = transformers.CLIPProcessor.from_pretrained(processor_name)

    def __repr__(self) -> str:
        return "CLIPModel()"

    @property
    def max_seq_length(self) -> int:
        return self.processor.tokenizer.model_max_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        self.processor.tokenizer.model_max_length = value

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image_embeds = []
        text_embeds = []

        if "pixel_values" in features:
            vision_outputs = self.model.vision_model(pixel_values=features["pixel_values"])
            image_embeds = self.model.visual_projection(vision_outputs[1])

        if "input_ids" in features:
            text_outputs = self.model.text_model(
                input_ids=features.get("input_ids"),
                attention_mask=features.get("attention_mask", None),
                position_ids=features.get("position_ids", None),
                output_attentions=features.get("output_attentions", None),
                output_hidden_states=features.get("output_hidden_states", None),
            )
            text_embeds = self.model.text_projection(text_outputs[1])

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

    def tokenize(self, texts, padding: str | bool = True) -> dict[str, torch.Tensor]:
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
            encoding = self.processor.tokenizer(texts_values, padding=padding, truncation=True, return_tensors="pt")

        if len(images):
            image_features = self.processor.image_processor(images, return_tensors="pt")
            encoding["pixel_values"] = image_features.pixel_values

        encoding["image_text_info"] = image_text_info
        return dict(encoding)

    @property
    def tokenizer(self) -> transformers.CLIPProcessor:
        return self.processor

    def save(self, output_path: str) -> None:
        self.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)

    @staticmethod
    def load(input_path: str) -> CLIPModel:
        return CLIPModel(model_name=input_path)
