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
"""Gpt2 preprocessing layers."""

import copy
import os

from tensorflow import keras

from keras_nlp.layers.start_end_packer import StartEndPacker
from keras_nlp.models.gpt2.gpt2_presets import backbone_presets
from keras_nlp.models.utils import classproperty
from keras_nlp.models.utils import pack_x_y_sample_weight
from keras_nlp.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras.utils.register_keras_serializable(package="keras_nlp")
class Gpt2Tokenizer(BytePairTokenizer):
    def __init__(
        self,
        vocabulary,
        merges,
        **kwargs,
    ):
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )

        # Check for necessary special tokens.
        start_token = "<|endoftext|>"
        for token in [start_token]:
            if token not in self.get_vocabulary():
                raise ValueError(
                    f"Cannot find token `'{token}'` in the provided "
                    f"`vocabulary`. Please provide `'{token}'` in your "
                    "`vocabulary` or use a pretrained `vocabulary` name."
                )

        self.start_token_id = self.token_to_id(start_token)

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
            "vocab.json",
            metadata["vocabulary_url"],
            cache_subdir=os.path.join("models", preset),
            file_hash=metadata["vocabulary_hash"],
        )
        merges = keras.utils.get_file(
            "merges.txt",
            metadata["merges_url"],
            cache_subdir=os.path.join("models", preset),
            file_hash=metadata["merges_hash"],
        )

        config = metadata["preprocessor_config"]
        config.update(
            {
                "vocabulary": vocabulary,
                "merges": merges,
            },
        )

        return cls.from_config({**config, **kwargs})


@keras.utils.register_keras_serializable(package="keras_nlp")
class Gpt2TextGeneratorPreprocessor(keras.layers.Layer):
    def __init__(
        self,
        tokenizer="gpt2_base",
        sequence_length=512,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Load backbone from string identifier
        if isinstance(tokenizer, str):
            tokenizer = Gpt2Tokenizer.from_preset(tokenizer)

        self.tokenizer = tokenizer
        self.packer = StartEndPacker(
            start_value=self.tokenizer.start_token_id,
            sequence_length=sequence_length + 1,
        )

    def vocabulary_size(self):
        """Returns the vocabulary size of the tokenizer."""
        return self.tokenizer.vocabulary_size()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "tokenizer": keras.layers.serialize(self.tokenizer),
                "sequence_length": self.packer.sequence_length - 1,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if "tokenizer" in config:
            config["tokenizer"] = keras.layers.deserialize(config["tokenizer"])
        return cls(**config)

    def call(self, x, y=None, sample_weight=None, return_targets=True):
        token_ids = self.tokenizer(x)
        token_ids = self.packer(token_ids)
        x = {
            "token_ids": token_ids[:, :-1],
            "padding_mask": token_ids[:, :-1] != 0,  # TODO fix.
        }
        if return_targets:
            y = token_ids[:, 1:]
            sample_weight = token_ids[:, :-1] != 0
            return pack_x_y_sample_weight(x, y, sample_weight)
        else:
            return x
