from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)


def cross_encoder_init_args_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        kwargs_renamed_mapping = {
            "model_name": "model_name_or_path",
            "automodel_args": "model_kwargs",
            "tokenizer_args": "tokenizer_kwargs",
            "config_args": "config_kwargs",
            "cache_dir": "cache_folder",
            "default_activation_function": "activation_fn",
        }
        for old_name, new_name in kwargs_renamed_mapping.items():
            if old_name in kwargs:
                kwarg_value = kwargs.pop(old_name)
                logger.warning(
                    f"The CrossEncoder `{old_name}` argument was renamed and is now deprecated, please use `{new_name}` instead."
                )
                if new_name not in kwargs:
                    kwargs[new_name] = kwarg_value

        if "classifier_dropout" in kwargs:
            classifier_dropout = kwargs.pop("classifier_dropout")
            logger.warning(
                f"The CrossEncoder `classifier_dropout` argument is deprecated. Please use `config_kwargs={{'classifier_dropout': {classifier_dropout}}}` instead."
            )
            if "config_kwargs" not in kwargs:
                kwargs["config_kwargs"] = {"classifier_dropout": classifier_dropout}
            else:
                kwargs["config_kwargs"]["classifier_dropout"] = classifier_dropout

        return func(self, *args, **kwargs)

    return wrapper


def cross_encoder_predict_rank_args_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        kwargs_renamed_mapping = {
            "activation_fct": "activation_fn",
        }
        for old_name, new_name in kwargs_renamed_mapping.items():
            if old_name in kwargs:
                kwarg_value = kwargs.pop(old_name)
                logger.warning(
                    f"The CrossEncoder.predict `{old_name}` argument was renamed and is now deprecated, please use `{new_name}` instead."
                )
                if new_name not in kwargs:
                    kwargs[new_name] = kwarg_value

        deprecated_args = ["num_workers"]

        for deprecated_arg in deprecated_args:
            if deprecated_arg in kwargs:
                kwargs.pop(deprecated_arg)
                logger.warning(
                    f"The CrossEncoder.predict `{deprecated_arg}` argument is deprecated and has no effect. It will be removed in a future version."
                )

        return func(self, *args, **kwargs)

    return wrapper
