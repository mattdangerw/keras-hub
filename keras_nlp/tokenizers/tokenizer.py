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

from typing import List

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.layers.preprocessing.preprocessing_layer import (
    PreprocessingLayer,
)
from keras_nlp.utils.preset_utils import check_config_class
from keras_nlp.utils.preset_utils import list_presets
from keras_nlp.utils.preset_utils import load_from_preset
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring


@keras_nlp_export(
    [
        "keras_nlp.models.Tokenizer",
        "keras_nlp.tokenizers.Tokenizer",
    ]
)
class Tokenizer(PreprocessingLayer):
    """A base class for tokenizer layers.

    Tokenizers in the KerasNLP library should all subclass this layer.
    The class provides two core methods `tokenize()` and `detokenize()` for
    going from plain text to sequences and back. A tokenizer is a subclass of
    `keras.layers.Layer` and can be combined into a `keras.Model`.

    Subclassers should always implement the `tokenize()` method, which will also
    be the default when calling the layer directly on inputs.

    Subclassers can optionally implement the `detokenize()` method if the
    tokenization is reversible. Otherwise, this can be skipped.

    Subclassers should implement `get_vocabulary()`, `vocabulary_size()`,
    `token_to_id()` and `id_to_token()` if applicable. For some simple
    "vocab free" tokenizers, such as a whitespace splitter show below, these
    methods do not apply and can be skipped.

    Example:

    ```python
    class WhitespaceSplitterTokenizer(keras_nlp.tokenizers.Tokenizer):
        def tokenize(self, inputs):
            return tf.strings.split(inputs)

        def detokenize(self, inputs):
            return tf.strings.reduce_join(inputs, separator=" ", axis=-1)

    tokenizer = WhitespaceSplitterTokenizer()

    # Tokenize some inputs.
    tokenizer.tokenize("This is a test")

    # Shorthard for `tokenize()`.
    tokenizer("This is a test")

    # Detokenize some outputs.
    tokenizer.detokenize(["This", "is", "a", "test"])
    ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tokenize(self, inputs, *args, **kwargs):
        """Transform input tensors of strings into output tokens.

        Args:
            inputs: Input tensor, or dict/list/tuple of input tensors.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError(
            "No implementation of `tokenize()` was found for "
            f"{self.__class__.__name__}. All tokenizers should implement "
            "`tokenize()`."
        )

    def detokenize(self, inputs, *args, **kwargs):
        """Transform tokens back into strings.

        Args:
            inputs: Input tensor, or dict/list/tuple of input tensors.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError(
            "No implementation of `detokenize()` was found for "
            f"{self.__class__.__name__}."
        )

    def get_vocabulary(self) -> List[str]:
        """Get the tokenizer vocabulary as a list of strings terms."""
        raise NotImplementedError(
            "No implementation of `get_vocabulary()` was found for "
            f"{self.__class__.__name__}."
        )

    def vocabulary_size(self) -> int:
        """Returns the total size of the token id space."""
        raise NotImplementedError(
            "No implementation of `vocabulary_size()` was found for "
            f"{self.__class__.__name__}."
        )

    def id_to_token(self, id: int) -> str:
        """Convert an integer id to a string token."""
        raise NotImplementedError(
            "No implementation of `id_to_token()` was found for "
            f"{self.__class__.__name__}."
        )

    def token_to_id(self, token: str) -> int:
        """Convert a string token to an integer id."""
        raise NotImplementedError(
            "No implementation of `token_to_id()` was found for "
            f"{self.__class__.__name__}."
        )

    def call(self, inputs, *args, training=None, **kwargs):
        return self.tokenize(inputs, *args, **kwargs)

    @classproperty
    def presets(cls):
        """List builtin presets for a `Task` subclass."""
        # We can also load backbone presets.
        return list_presets(cls)

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        """Instantiate {{model_name}} tokenizer from preset vocabulary.

        Args:
            preset: string. Must be one of "{{preset_names}}".

        Examples:
        ```python
        # Load a preset tokenizer.
        tokenizer = {{model_name}}.from_preset("{{example_preset_name}}")

        # Tokenize some input.
        tokenizer("The quick brown fox tripped.")

        # Detokenize some input.
        tokenizer.detokenize([5, 6, 7, 8, 9])
        ```
        """
        config_file = "tokenizer.json"
        preset_cls = check_config_class(preset, config_file=config_file)
        if not issubclass(preset_cls, cls):
            raise ValueError(
                f"Preset has type `{preset_cls.__name__}` which is not a "
                f"a subclass of calling class `{cls.__name__}`. Call "
                f"`from_preset` directly on `{preset_cls.__name__}` instead."
            )
        return load_from_preset(
            preset,
            config_file=config_file,
            config_overrides=kwargs,
        )

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
            cls.from_preset.__func__.__doc__ = Tokenizer.from_preset.__doc__
            format_docstring(
                model_name=cls.__name__,
                example_preset_name=next(iter(cls.presets), ""),
                preset_names='", "'.join(cls.presets),
            )(cls.from_preset.__func__)
