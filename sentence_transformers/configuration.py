import logging
from typing import Optional, List, Dict, Any, Union
import transformers
from sentence_transformers import __version__
import torch
import os

from transformers import PreTrainedModel, AutoModel, PretrainedConfig, AutoConfig, BertModel, AutoTokenizer, PreTrainedTokenizer
from transformers.utils import cached_file
from transformers.dynamic_module_utils import custom_object_save
from os import PathLike

logger = logging.getLogger(__name__)

BASIC_MODELS = ['albert-base-v1', 'albert-base-v2', 'albert-large-v1', 'albert-large-v2', 'albert-xlarge-v1', 'albert-xlarge-v2', 'albert-xxlarge-v1', 'albert-xxlarge-v2', 'bert-base-cased-finetuned-mrpc', 'bert-base-cased', 'bert-base-chinese', 'bert-base-german-cased', 'bert-base-german-dbmdz-cased', 'bert-base-german-dbmdz-uncased', 'bert-base-multilingual-cased', 'bert-base-multilingual-uncased', 'bert-base-uncased', 'bert-large-cased-whole-word-masking-finetuned-squad', 'bert-large-cased-whole-word-masking', 'bert-large-cased', 'bert-large-uncased-whole-word-masking-finetuned-squad', 'bert-large-uncased-whole-word-masking', 'bert-large-uncased', 'camembert-base', 'ctrl', 'distilbert-base-cased-distilled-squad', 'distilbert-base-cased', 'distilbert-base-german-cased', 'distilbert-base-multilingual-cased', 'distilbert-base-uncased-distilled-squad', 'distilbert-base-uncased-finetuned-sst-2-english', 'distilbert-base-uncased', 'distilgpt2', 'distilroberta-base', 'gpt2-large', 'gpt2-medium', 'gpt2-xl', 'gpt2', 'openai-gpt', 'roberta-base-openai-detector', 'roberta-base', 'roberta-large-mnli', 'roberta-large-openai-detector', 'roberta-large', 't5-11b', 't5-3b', 't5-base', 't5-large', 't5-small', 'transfo-xl-wt103', 'xlm-clm-ende-1024', 'xlm-clm-enfr-1024', 'xlm-mlm-100-1280', 'xlm-mlm-17-1280', 'xlm-mlm-en-2048', 'xlm-mlm-ende-1024', 'xlm-mlm-enfr-1024', 'xlm-mlm-enro-1024', 'xlm-mlm-tlm-xnli15-1024', 'xlm-mlm-xnli15-1024', 'xlm-roberta-base', 'xlm-roberta-large-finetuned-conll02-dutch', 'xlm-roberta-large-finetuned-conll02-spanish', 'xlm-roberta-large-finetuned-conll03-english', 'xlm-roberta-large-finetuned-conll03-german', 'xlm-roberta-large', 'xlnet-base-cased', 'xlnet-large-cased']

class SentenceTransformerConfig(PretrainedConfig):
    model_type = "sentence-transformer"
    is_composition = True

    def __init__(
        self,
        # encoder_config: Optional[PretrainedConfig] = None,
        **kwargs,
    ) -> None:
        # self.encoder = encoder_config
        self.max_seq_length: Optional[int] = kwargs.pop("max_seq_length", None)
        self.do_lower_case: bool = kwargs.pop("do_lower_case", False)
        self.modules: List[str] = kwargs.pop("modules", [])
        # TODO: Duplicate from transformers_version
        self.__version__: Dict[str, str] = kwargs.pop("__version__", {
                'sentence_transformers': __version__,
                'transformers': transformers.__version__,
                'pytorch': torch.__version__,
            })
        super().__init__(**kwargs)

    # def __getattribute__(self, key: str) -> Any:
    #     try:
    #         return super().__getattribute__(key)
    #     except AttributeError as e:
    #         try:
    #             return super().__getattribute__("encoder").__getattribute__(key)
    #         except KeyError:
    #             raise e

    # def save_pretrained(self, save_directory: str | PathLike, push_to_hub: bool = False, **kwargs) -> None:
    #     original_config_name = transformers.utils.CONFIG_NAME
    #     transformers.utils.CONFIG_NAME = "config_sentence_transformers.json"
    #     super().save_pretrained(save_directory, push_to_hub, **kwargs)
    #     transformers.utils.CONFIG_NAME = original_config_name

    # def to_dict(self) -> Dict[str, Any]:
    #     class_dict = super().to_dict()
    #     class_dict.pop("encoder", None)
    #     return class_dict

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~PretrainedConfig.from_pretrained`] class method.

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
        # self.encoder.save_pretrained(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)
        # TODO: Maybe just save this first, then move to "config_sentence_transformers.json"?

        self._set_token_in_kwargs(kwargs)

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, "config_sentence_transformers.json")

        self.to_json_file(output_config_file, use_diff=True)
        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

    # @classmethod
    # def from_pretrained(
    #     cls,
    #     pretrained_model_name_or_path: str | PathLike,
    #     cache_dir: str | PathLike | None = None,
    #     force_download: bool = False,
    #     local_files_only: bool = False,
    #     token: str | bool | None = None,
    #     revision: str = "main",
    #     **kwargs
    # ) -> PretrainedConfig:
    #     return super().from_pretrained(pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, _configuration_file="sentence_bert_config.json", **kwargs)
