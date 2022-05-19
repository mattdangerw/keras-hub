# Copyright 2022 The KerasNLP Authors
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

import tensorflow as tf
from tensorflow import keras

from keras_nlp.utils import tensor_utils


class Tokenizer(keras.layers.Layer):
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

    Examples:

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

    def __new__(cls, *args, **kwargs):
        # Wrap the `tokenize` and `detokenize` methods so they route through
        # __call__. This is needed for functional model support.
        obj = super().__new__(cls, *args, **kwargs)
        obj._subclass_tokenize = obj.tokenize
        obj._subclass_detokenize = obj.detokenize
        obj.tokenize = obj._wrapped_tokenize
        obj.detokenize = obj._wrapped_detokenize
        return obj

    def tokenize(self, inputs):
        """Transform input tensors of strings into output tokens.

        Args:
            inputs: Input tensor, or dict/list/tuple of input tensors.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError(
            "No implementation of `tokenize()` was found for "
            f"{self.__class__.__name__}. All tokenizers should implement "
            "`tokenize()`."
        )

    def detokenize(self, inputs, ids_to_strip=[0], return_stings=True):
        """Transform tokens back into strings.

        Args:
            inputs: Input tensor, or dict/list/tuple of input tensors.
            ids_to_strip: Token ids to strip before detokenizing. Defaults to
                `[0]` to strip padding tokens.
            return_stings: If true, return decoded lists of python strings
                rather than tensors of encoded byte strings.
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
        """Convert an integer id to a string token."""
        raise NotImplementedError(
            "No implementation of `id_to_token()` was found for "
            f"{self.__class__.__name__}."
        )

    def _wrapped_tokenize(self, *args, **kwargs):
        """The wrapped tokenize method which routes through call."""
        return self(*args, mode="tokenize", **kwargs)

    def _wrapped_detokenize(self, *args, **kwargs):
        """The wrapped detokenize method which routes through call."""
        return self(*args, mode="detokenize", **kwargs)

    def _base_tokenize(self, inputs):
        """Common tokenization logic to all tokenizers."""
        if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            inputs = tf.convert_to_tensor(inputs)
        return self._subclass_tokenize(inputs)

    def _base_detokenize(self, inputs, ids_to_strip=[0], return_stings=False):
        """Common detokenization logic to all tokenizers."""
        if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            inputs = tf.convert_to_tensor(inputs)

        if ids_to_strip and inputs.dtype.is_integer:
            # Strip ids we don't want for detokenization.
            mask = tf.ones_like(inputs, dtype=tf.bool)
            for id in ids_to_strip:
                mask = tf.logical_and(mask, inputs != id)
            inputs = tf.ragged.boolean_mask(inputs, mask)

        outputs = self._subclass_detokenize(inputs)

        if return_stings:
            outputs = tensor_utils.tensor_to_string_list(outputs)
        return outputs

    def call(self, *args, mode="tokenize", training=None, **kwargs):
        if mode == "tokenize":
            return self._base_tokenize(*args, **kwargs)
        elif mode == "detokenize":
            return self._base_detokenize(*args, **kwargs)
        else:
            raise ValueError(
                f"Unsupported tokenizer mode. Received: mode={mode}"
            )
