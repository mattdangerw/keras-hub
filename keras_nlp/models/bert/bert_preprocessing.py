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
"""BERT preprocessing layers."""

import copy
import os

from tensorflow import keras

from keras_nlp.layers.multi_segment_packer import MultiSegmentPacker
from keras_nlp.models.bert.bert_presets import backbone_presets
from keras_nlp.models.utils import classproperty
from keras_nlp.tokenizers.word_piece_tokenizer import WordPieceTokenizer
from keras_nlp.utils.keras_utils import pack_x_y_sample_weight


@keras.utils.register_keras_serializable(package="keras_nlp")
class BertTokenizer(WordPieceTokenizer):
    def __init__(
        self,
        vocabulary,
        lowercase=False,
        **kwargs,
    ):
        super().__init__(
            vocabulary=vocabulary,
            lowercase=lowercase,
            **kwargs,
        )

        # Check for necessary special tokens.
        for token in ["[CLS]", "[SEP]", "[PAD]", "[MASK]"]:
            if token not in self.get_vocabulary():
                raise ValueError(
                    f"Cannot find token `'{token}'` in the provided "
                    f"`vocabulary`. Please provide `'{token}'` in your "
                    "`vocabulary` or use a pretrained `vocabulary` name."
                )

        # Special token ids. Expose these as properties?
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        self.cls_token_id = self.token_to_id(self.cls_token)
        self.sep_token_id = self.token_to_id(self.sep_token)
        self.pad_token_id = self.token_to_id(self.pad_token)
        self.mask_token_id = self.token_to_id(self.mask_token)

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        if preset not in cls.presets:
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}."""
            )
        metadata = cls.presets[preset]

        vocabulary = keras.utils.get_file(
            "vocab.txt",
            metadata["vocabulary_url"],
            cache_subdir=os.path.join("models", preset),
            file_hash=metadata["vocabulary_hash"],
        )

        config = metadata["preprocessor_config"]
        config.update(
            {
                "vocabulary": vocabulary,
            },
        )

        return cls.from_config({**config, **kwargs})


@keras.utils.register_keras_serializable(package="keras_nlp")
class BertPreprocessor(keras.layers.Layer):
    """A BERT preprocessing layer which tokenizes and packs inputs.

    This preprocessing layer will do three things:

     - Tokenize any number of inputs using the `tokenizer`.
     - Pack the inputs together using a `keras_nlp.layers.MultiSegmentPacker`.
       with the appropriate `"[CLS]"`, `"[SEP]"` and `"[PAD]"` tokens.
     - Construct a dictionary of with keys `"token_ids"`, `"segment_ids"`,
       `"padding_mask"`, that can be passed directly to a BERT model.

    Args:
        tokenizer: A `BertTokenizer`.
        sequence_length: The length of the packed inputs.
        truncate: string. The algorithm to truncate a list of batched segments
            to fit within `sequence_length`. The value can be either
            `round_robin` or `waterfall`:
                - `"round_robin"`: Available space is assigned one token at a
                    time in a round-robin fashion to the inputs that still need
                    some, until the limit is reached.
                - `"waterfall"`: The allocation of the budget is done using a
                    "waterfall" algorithm that allocates quota in a
                    left-to-right manner and fills up the buckets until we run
                    out of budget. It supports an arbitrary number of segments.

    Examples:
    ```python
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    vocab += ["the", "qu", "##ick", "br", "##own", "fox", "tripped"]
    vocab += ["call", "me", "ish", "##mael", "."]
    vocab += ["oh", "look", "a", "whale"]
    vocab += ["i", "forgot", "my", "home", "##work"]
    preprocessor = keras_nlp.models.BertPreprocessor(vocabulary=vocab)

    # Tokenize and pack a single sentence directly.
    preprocessor("The quick brown fox jumped.")

    # Tokenize and pack a multiple sentence directly.
    preprocessor(("The quick brown fox jumped.", "Call me Ishmael."))

    # Map a dataset to preprocess a single sentence.
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    labels = [0, 1]
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Map a dataset to preprocess a multiple sentences.
    first_sentences = ["The quick brown fox jumped.", "Call me Ishmael."]
    second_sentences = ["The fox tripped.", "Oh look, a whale."]
    labels = [1, 1]
    ds = tf.data.Dataset.from_tensor_slices(
        (
            (first_sentences, second_sentences), labels
        )
    )
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    def __init__(
        self,
        tokenizer="bert_base_uncased_en",
        sequence_length=512,
        truncate="round_robin",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.cls_token_id,
            end_value=self.tokenizer.sep_token_id,
            pad_value=self.tokenizer.pad_token_id,
            truncate=truncate,
            sequence_length=sequence_length,
        )

    def vocabulary_size(self):
        """Returns the vocabulary size of the tokenizer."""
        return self.tokenizer.vocabulary_size()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "tokenizer": keras.layers.serialize(self.tokenizer),
                "sequence_length": self.packer.sequence_length,
                "truncate": self.packer.truncate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if "tokenizer" in config:
            config["tokenizer"] = keras.layers.deserialize(config["tokenizer"])
        return cls(**config)

    def call(self, x, y=None, sample_weight=None):
        if not isinstance(x, (list, tuple)):
            x = [x]

        segments = [self.tokenizer(segment) for segment in x]
        token_ids, segment_ids = self.packer(segments)
        x = {
            "token_ids": token_ids,
            "segment_ids": segment_ids,
            "padding_mask": token_ids != self.tokenizer.pad_token_id,
        }
        return pack_x_y_sample_weight(x, y, sample_weight)

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        if "tokenizer" not in kwargs:
            kwargs["tokenizer"] = BertTokenizer.from_preset(preset)

        # Use model's `max_sequence_length` if `sequence_length` unspecified;
        # otherwise check that `sequence_length` not too long.
        metadata = cls.presets[preset]
        max_sequence_length = metadata["config"]["max_sequence_length"]
        if "sequence_length" not in kwargs:
            kwargs["sequence_length"] = max_sequence_length
        else:
            if kwargs["sequence_length"] > max_sequence_length:
                raise ValueError(
                    f"`sequence_length` cannot be longer than `{preset}` "
                    f"preset's `max_sequence_length` of {max_sequence_length}. "
                    f"Received: {kwargs['sequence_length']}."
                )

        return cls(**kwargs)


FROM_PRESET_DOCSTRING = """Instantiate BERT preprocessor from preset architecture.

    Args:
        preset: string. Must be one of {names}.
        sequence_length: int, optional. The length of the packed inputs. Must be
            equal to or smaller than the `max_sequence_length` of the preset. If
            left as default, the `max_sequence_length` of the preset will be
            used.
        truncate: string. The algorithm to truncate a list of batched segments
            to fit within `sequence_length`. The value can be either
            `round_robin` or `waterfall`:
                - `"round_robin"`: Available space is assigned one token at a
                    time in a round-robin fashion to the inputs that still need
                    some, until the limit is reached.
                - `"waterfall"`: The allocation of the budget is done using a
                    "waterfall" algorithm that allocates quota in a
                    left-to-right manner and fills up the buckets until we run
                    out of budget. It supports an arbitrary number of segments.

    Examples:
    ```python
    # Load preprocessor from preset
    preprocessor = BertPreprocessor.from_preset("bert_base_uncased_en")
    preprocessor("The quick brown fox jumped.")

    # Override sequence_length
    preprocessor = BertPreprocessor.from_preset(
        "bert_base_uncased_en"
        sequence_length=64
    )
    preprocessor("The quick brown fox jumped.")
    ```
    """

setattr(
    BertPreprocessor.from_preset.__func__,
    "__doc__",
    FROM_PRESET_DOCSTRING.format(names=", ".join(BertPreprocessor.presets)),
)
