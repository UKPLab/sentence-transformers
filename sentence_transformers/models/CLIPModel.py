from torch import nn
import transformers
import torch
from PIL import Image


class CLIPModel(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", processor_name=None):
        super(CLIPModel, self).__init__()

        if processor_name is None:
            processor_name = model_name

        self.model = transformers.CLIPModel.from_pretrained(model_name)
        self.processor = transformers.CLIPProcessor.from_pretrained(processor_name)

    def __repr__(self):
        return "CLIPModel()"

    def forward(self, features):
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

    def tokenize(self, texts):
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

        if len(texts_values) == 0:
            texts_values = None
        if len(images) == 0:
            images = None

        inputs = self.processor(text=texts_values, images=images, return_tensors="pt", padding=True)
        inputs["image_text_info"] = image_text_info
        return inputs

    def save(self, output_path: str):
        self.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)

    @staticmethod
    def load(input_path: str):
        return CLIPModel(model_name=input_path)
