import logging
from transformers.integrations.peft import PeftAdapterMixin as PeftAdapterMixinTransformers
logger = logging.getLogger(__name__)


class PeftAdapterMixin:
    """
    Wrapper Mixin that adds the functionality to easily load and use adapters on the model. For
    more details about adapters check out the documentation of PEFT
    library: https://huggingface.co/docs/peft/index

    Currently supported PEFT methods follow those supported by transformers library, 
    you can find more information on:
    https://huggingface.co/docs/transformers/main/en/peft#transformers.integrations.PeftAdapterMixin
    """

    def load_adapter(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
        Load adapter weights from file or remote Hub folder." If you are not familiar with adapters and PEFT methods, we
        invite you to read more about them on PEFT official documentation: https://huggingface.co/docs/peft

        Requires peft as a backend to load the adapter weights.

        Args:
            *args:
                Positional arguments to pass to the underlying AutoModel `load_adapter` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/peft#transformers.integrations.PeftAdapterMixin.load_adapter
            **kwargs:
                Keyword arguments to pass to the underlying AutoModel `load_adapter` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/peft#transformers.integrations.PeftAdapterMixin.load_adapter
        """
        self._modules["0"].auto_model.load_adapter(*args, **kwargs)

    def add_adapter(self, *args, **kwargs) -> None:
        """
        Adds a fresh new adapter to the current model for training purpose. If no adapter name is passed, a default
        name is assigned to the adapter to follow the convention of PEFT library (in PEFT we use "default" as the
        default adapter name).

        Args:
            *args:
                Positional arguments to pass to the underlying AutoModel `add_adapter` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/peft#transformers.integrations.PeftAdapterMixin.add_adapter
            **kwargs:
                Keyword arguments to pass to the underlying AutoModel `add_adapter` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/peft#transformers.integrations.PeftAdapterMixin.add_adapter
            
        """
        self._modules["0"].auto_model.add_adapter(*args, **kwargs)

    def set_adapter(self, *args, **kwargs) -> None:
        """
        Sets a specific adapter by forcing the model to use a that adapter and disable the other adapters.

        Args:
            *args:
                Positional arguments to pass to the underlying AutoModel `set_adapter` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/peft#transformers.integrations.PeftAdapterMixin.set_adapter
            **kwargs:
                Keyword arguments to pass to the underlying AutoModel `set_adapter` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/peft#transformers.integrations.PeftAdapterMixin.set_adapter
        """
        self._modules["0"].auto_model.set_adapter(*args, **kwargs)

    def disable_adapters(self) -> None:
        """
        Disable all adapters that are attached to the model. This leads to inferring with the base model only.
        """
        self._modules["0"].auto_model.disable_adapters()
        

    def enable_adapters(self) -> None:
        """
        Enable adapters that are attached to the model. The model will use `self.active_adapter()`
        """
        self._modules["0"].auto_model.enable_adapters()

    def active_adapters(self) -> list[str]:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Gets the current active adapters of the model. In case of multi-adapter inference (combining multiple adapters
        for inference) returns the list of all active adapters so that users can deal with them accordingly.

        For previous PEFT versions (that does not support multi-adapter inference), `module.active_adapter` will return
        a single string.
        """
        return self._modules["0"].auto_model.active_adapters()

    def active_adapter(self) -> str:
        return self._modules["0"].auto_model.active_adapter()

    def get_adapter_state_dict(self, *args, **kwargs) -> dict:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Gets the adapter state dict that should only contain the weights tensors of the specified adapter_name adapter.
        If no adapter_name is passed, the active adapter is used.

        Args:
            *args:
                Positional arguments to pass to the underlying AutoModel `get_adapter_state_dict` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/peft#transformers.integrations.PeftAdapterMixin.get_adapter_state_dict
            **kwargs:
                Keyword arguments to pass to the underlying AutoModel `get_adapter_state_dict` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/peft#transformers.integrations.PeftAdapterMixin.get_adapter_state_dict
        """
        return self._modules["0"].auto_model.get_adapter_state_dict(*args, **kwargs)