from collections import OrderedDict
import logging
from typing import Optional, List, Dict, Union
import transformers
from sentence_transformers import __version__
import torch
import os
from packaging.version import Version

from transformers import PretrainedConfig


logger = logging.getLogger(__name__)


class SentenceTransformerConfig(PretrainedConfig):
    model_type = "sentence-transformers"
    is_composition = True

    def __init__(self, **kwargs) -> None:
        self.modules: List[str] = kwargs.pop("modules", [])
        self.__version__: Dict[str, str] = kwargs.pop(
            "__version__",
            {
                "sentence_transformers": __version__,
                "transformers": transformers.__version__,
                "pytorch": torch.__version__,
            },
        )
        super().__init__(**kwargs)

        if Version(self.__version__["sentence_transformers"]) > Version(__version__):
            logger.warning(
                f"You try to use a model that was created with version {self.__version__['sentence_transformers']}, "
                f"however, your version is {__version__}. This might cause unexpected behavior or errors. "
                "In that case, try to update to the latest version."
            )

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~SentenceTransformerConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        from transformers import configuration_utils

        # Override the config name to make sure we save this as a SentenceTransformer config
        original_config_name = configuration_utils.CONFIG_NAME[:]
        configuration_utils.CONFIG_NAME = "config_sentence_transformers.json"
        super().save_pretrained(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)
        configuration_utils.CONFIG_NAME = original_config_name

    @classmethod
    def from_modules(cls, modules: Optional[OrderedDict]) -> "SentenceTransformerConfig":
        """
        Creates a SentenceTransformerConfig from a list of modules.
        """
        kwargs = {}
        module_configs = []
        for module in modules.values():
            module_configs.append(
                {
                    "type": type(module).__module__,
                    "config": module.get_config_dict() if hasattr(module, "get_config_dict") else {},
                }
            )
        kwargs["modules"] = module_configs
        return cls(**kwargs)
