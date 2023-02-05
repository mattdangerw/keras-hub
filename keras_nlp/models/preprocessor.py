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

from tensorflow import keras

from keras_nlp.utils.keras_utils import deserialize_preset
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring


@keras.utils.register_keras_serializable(package="keras_nlp")
class Preprocessor(keras.layers.Layer):
    """Base class for model preprocessors."""

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
        return {}

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        """Instantiate {{preprocessor_name}} from preset architecture.

        Args:
            preset: string. Must be one of "{{preset_names}}".

        Examples:
        ```python
        # Load preprocessor from preset
        preprocessor = keras_nlp.models.{{preprocessor_name}}.from_preset(
            "{{example_preset_name}}",
        )
        preprocessor("The quick brown fox jumped.")

        # Override sequence_length
        preprocessor = keras_nlp.models.{{preprocessor_name}}.from_preset(
            "{{example_preset_name}}",
            sequence_length=64
        )
        preprocessor("The quick brown fox jumped.")
        ```
        """
        if not cls.presets:
            raise NotImplementedError(
                "No presets have been created for this class."
            )

        if preset not in cls.presets:
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}."""
            )

        if "sequence_length" in kwargs:
            max = cls.presets[preset]["sequence_length"]
            value = kwargs["sequence_length"]
            if value > max:
                raise ValueError(
                    f"`sequence_length` cannot be longer than the maxiumum "
                    "sequence length supported by preset `'{preset}'`, "
                    f"{max}. Received: {value}."
                )

        config = {**cls.presets[preset], **kwargs}
        return deserialize_preset(cls, preset, config)

    def __init_subclass__(cls, **kwargs):
        # Use __init_subclass__ to setup a correct docstring for from_preset.
        super().__init_subclass__(**kwargs)

        # If the subclass does not define from_preset, assign a wrapper so that
        # each class can have an distinct docstring.
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
