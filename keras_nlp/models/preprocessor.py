# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.backend import keras
from keras_nlp.layers.preprocessing.preprocessing_layer import (
    PreprocessingLayer,
)
from keras_nlp.utils.preset_utils import check_config_class
from keras_nlp.utils.preset_utils import get_registered_presets
from keras_nlp.utils.preset_utils import get_registered_subclasses
from keras_nlp.utils.preset_utils import load_from_preset
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring


@keras_nlp_export("keras_nlp.models.Preprocessor")
class Preprocessor(PreprocessingLayer):
    """Base class for preprocessing layers.

    A `Preprocessor` layer wraps a `keras_nlp.tokenizer.Tokenizer` to provide a
    complete preprocessing setup for a given task. For example a masked language
    modeling preprocessor will take in raw input strings, and output
    `(x, y, sample_weight)` tuples. Where `x` contains token id sequences with
    some

    This class can be subclassed to implement
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = None

    def __setattr__(self, name, value):
        # Work around torch setattr for properties.
        if name in ["tokenizer"]:
            return object.__setattr__(self, name, value)
        return super().__setattr__(name, value)

    @property
    def tokenizer(self):
        """The tokenizer used to tokenize strings."""
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        self._tokenizer = value

    def get_config(self):
        config = super().get_config()
        config["tokenizer"] = keras.layers.serialize(self.tokenizer)
        return config

    @classmethod
    def from_config(cls, config):
        if "tokenizer" in config and isinstance(config["tokenizer"], dict):
            config["tokenizer"] = keras.layers.deserialize(config["tokenizer"])
        return cls(**config)

    @classproperty
    def tokenizer_cls(cls):
        return None

    @classproperty
    def presets(cls):
        return get_registered_presets(cls)

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        """Instantiate {{preprocessor_name}} from preset architecture.

        Args:
            preset: string. Must be one of "{{preset_names}}".

        Example:
        ```python
        # Load a preprocessor layer from a preset.
        preprocessor = keras_nlp.models.{{preprocessor_name}}.from_preset(
            "{{example_preset_name}}",
        )
        ```
        """
        if cls == Preprocessor:
            raise ValueError(
                "Do not call `Preprocessor.from_preset()` directly. Instead call a "
                "choose a particular task class, e.g. "
                "`keras_nlp.models.BertPreprocessor.from_preset()`."
            )
        config_file = "tokenizer.json"
        preset_cls = check_config_class(preset, config_file=config_file)
        subclasses = get_registered_subclasses(cls)
        subclasses = tuple(
            filter(lambda x: x.backbone_cls == preset_cls, subclasses)
        )
        if len(subclasses) == 0:
            raise ValueError(
                f"No registered subclass of `{cls.__name__}` can load "
                f"a `Tokenizer` of class `{preset_cls.__name__}`. Try "
                f"`print({cls.__name__}.presets)` to see a list of allowed "
                "preset names."
            )
        if len(subclasses) > 2:
            raise ValueError(
                f"Ambiguous call to `{cls.__name__}.from_preset()`. Found "
                f"multiple registered subclasses {subclasses}. Please call "
                "`from_preset` on a subclass directly."
            )
        cls = subclasses[0]
        tokenizer = load_from_preset(
            preset,
            config_file=config_file,
        )
        return cls(tokenizer=tokenizer, **kwargs)

    def __init_subclass__(cls, **kwargs):
        # Use __init_subclass__ to setup a correct docstring for from_preset.
        super().__init_subclass__(**kwargs)

        # If the subclass does not define from_preset, assign a wrapper so that
        # each class can have a distinct docstring.
        if "from_preset" not in cls.__dict__:

            def from_preset(calling_cls, *args, **kwargs):
                return super(cls, calling_cls).from_preset(*args, **kwargs)

            cls.from_preset = classmethod(from_preset)

        # Format and assign the docstring unless the subclass has overridden it.
        if cls.from_preset.__doc__ is None:
            cls.from_preset.__func__.__doc__ = Preprocessor.from_preset.__doc__
            format_docstring(
                preprocessor_name=cls.__name__,
                example_preset_name=next(iter(cls.presets), ""),
                preset_names='", "'.join(cls.presets),
            )(cls.from_preset.__func__)
