import numpy as np
import logging
import os
from typing import Union
import torch

from .. import SentenceTransformer

logger = logging.getLogger(__name__)


class CT2SentenceTransformer(SentenceTransformer):
    """
    Loads or create a SentenceTransformer model, that can be used to map sentences / text to embeddings.
    Extension of sentence_transformers.SentenceTransformer using a CTranslate2 model for accelerated inference-only.
    Adapted from https://gist.github.com/guillaumekln/fb125fc3eb108d1a304b7432486e712f

    :param model_name_or_path: If it is a filepath on disc, it loads the model from that path. If it is not a path, it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model from Huggingface models repository with that name.
    :param modules: This parameter can be used to create custom SentenceTransformer models from scratch.
    :param device: Device (like 'cuda' / 'cpu') that should be used for computation. If None, checks if a GPU can be used.
    :param cache_folder: Path to store models. Can be also set by SENTENCE_TRANSFORMERS_HOME enviroment variable.
    :param use_auth_token: HuggingFace authentication token to download private models.
    :param compute_type: weight quantization, scheme for computing,
        (possible values are: int8, int8_float16, int16, float16).
    :param force: force new conversion with CTranslate2, even if it already exists.
    :param vmap: Optional path to a vocabulary mapping file that will be included
        in the converted model directory.
    """

    def __init__(
        self,
        *args,
        compute_type="default",
        force=False,
        vmap: Union[str,None] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self[0] = CT2Transformer(
            self[0],
            compute_type=compute_type,
            force=force,
            vmap=vmap,
        )


class CT2Transformer(torch.nn.Module):
    """Wrapper around a sentence_transformers.models.Transformer which routes the forward
    call to a CTranslate2 encoder model.

    :param compute_type: weight quantization, scheme for computing,
        default uses same as quantization
        (possible values are: int8, int8_float16, int16, float16).
    :param force: force new conversion with CTranslate2, even if it already exists.
    :param vmap: Optional path to a vocabulary mapping file that will be included
        in the converted model directory.
    """

    def __init__(
        self,
        transformer,
        compute_type="default",
        force=False,
        vmap: Union[str,None] = None,
    ):
        super().__init__()
        try:
            import ctranslate2
        except ImportError:
            logger.exception(
                "for running the CT2SentenceTransformer,"
                " it is required to install CTranslate2 by running "
                " `pip install ctranslate2>=3.16.0`"
            )

        self.tokenize = transformer.tokenize
        self.compute_type = compute_type
        self.encoder = None

        # Convert to the CTranslate2 model format, if not already done.
        model_dir = transformer.auto_model.config.name_or_path
        self.ct2_model_dir = os.path.join(
            model_dir,
            "ctranslate2_" + ctranslate2.__version__,
        )

        if not os.path.exists(os.path.join(self.ct2_model_dir, "model.bin")) or force:
            if os.path.exists(self.ct2_model_dir) and not os.listdir(
                self.ct2_model_dir
            ):
                force = True
            converter = ctranslate2.converters.TransformersConverter(model_dir)
            converter.convert(self.ct2_model_dir, force=force, vmap=vmap)
        self._ctranslate2_encoder_cls = ctranslate2.Encoder
        self._storage_view = ctranslate2.StorageView

    def children(self):
        # Do not consider the "transformer" attribute as a child module so that it will stay on the CPU.
        return []
    
    def half(self):
        self.to(dtype="float16")
        return self
        
    def to(self, device: str = "", dtype: str = ""):
        if not isinstance(device, str):
            raise ValueError("param `dtype` needs to be of type str")
        if not isinstance(dtype, str):
            raise ValueError("param `dtype` needs to be of type str")
        
        if dtype and not ("float" in dtype or "int" in dtype):
            raise ValueError("dtype should be one of `int8`, `float16`, `int8_float16`, `float32`")
        elif dtype:
            new_dtype = True
            self.compute_type = new_dtype
        else:
            new_dtype = False
        
        if device and (device.startswith("cuda") or device.startswith("cpu")):
            raise ValueError("for param `device`, f'cuda:{index}' or f'cpu:{index}' are supported")
        elif device:
            if ":" in device:
                new_device = device.split(":")[0]
                new_index = device.split(":")[1]
            else:
                new_device = device
                new_index = 0
        else:
            new_device = False
            new_index = False
            
        if new_device or new_dtype or new_index:        
            self.encoder = self._ctranslate2_encoder_cls(
                self.ct2_model_dir,
                device=new_device,
                device_index=new_index,
                intra_threads=torch.get_num_threads(),
                compute_type=self.compute_type,
            )
        return self
        

    def forward(self, features):
        """overwrites torch forward method with CTranslate model"""
        device = features["input_ids"].device

        if self.encoder is None:
            # The encoder is lazy-loaded to correctly resolve the target device.
            self.encoder = self._ctranslate2_encoder_cls(
                self.ct2_model_dir,
                device=device.type,
                device_index=device.index or 0,
                intra_threads=torch.get_num_threads(),
                compute_type=self.compute_type,
            )

        input_ids = features["input_ids"].to(torch.int32)
        length = features["attention_mask"].sum(1, dtype=torch.int32)

        if device.type == "cpu":
            # PyTorch CPU tensors do not implement the Array interface so a roundtrip to Numpy
            # is required for both the input and output.
            input_ids = input_ids.numpy()
            length = length.numpy()

        input_ids = self._storage_view.from_array(input_ids)
        length = self._storage_view.from_array(length)

        outputs = self.encoder.forward_batch(input_ids, length)

        last_hidden_state = outputs.last_hidden_state
        if device.type == "cpu":
            last_hidden_state = np.array(last_hidden_state)

        features["token_embeddings"] = torch.as_tensor(
            last_hidden_state, device=device
        ).to(torch.float32)

        return features

    def tokenize(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)
