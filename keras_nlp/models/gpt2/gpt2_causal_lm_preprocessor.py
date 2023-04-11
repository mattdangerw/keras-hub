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

"""GPT2 Causal LM preprocessor layer."""

import tensorflow as tf
from absl import logging

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.models.gpt2.gpt2_preprocessor import GPT2Preprocessor
from keras_nlp.utils.keras_utils import (
    convert_inputs_to_list_of_tensor_segments,
)
from keras_nlp.utils.keras_utils import pack_x_y_sample_weight
from keras_nlp.utils.tf_utils import tensor_to_string_list


@keras_nlp_export("keras_nlp.models.GPT2CausalLMPreprocessor")
class GPT2CausalLMPreprocessor(GPT2Preprocessor):
    """Preprocessor for `keras_nlp.models.GPT2CausalLM`.

    This preprocessing layer is meant for use with
    `keras_nlp.models.GPT2CausalLM`. It has three different modes of operation:

    1. `"training"`: Where a string input is tokenized and split into features
        and labels.
    2. `"generate_preprocessing"`: Where a string input is tokenized and
        packed for a `generate()` call.
    3. `"generate_postprocessing"`: Where a token id output is masked and
        detokenized after `generate()` has completed sampling new token ids.


    Args:
        tokenizer: A `keras_nlp.models.GPT2Tokenizer` instance.
        sequence_length: The length of the packed inputs.
        add_start_token: If true, the preprocessor will append the tokenizer
            start token to each input sequence.
        add_end_token: If true, the preprocessor will append the tokenizer
            end token to each input sequence.
        add_end_token_for_generate: If true, the preprocessor will also
            append an end token during generation.
        output_strings: If true, all tensors will be converted to python
            string types during generation postprocessing.
        output_special_tokens: If true, all special tokens will be preserved
            in the output during generation postprocessing.

    Call arguments:
        x: A string `tf.Tensor` or list of python strings.
        y: Label data. Should always be `None` as the layer generates labels.
        sample_weight: Label weights. Should always be `None` as the layer
            generates label weights.
        sequence_length: Pass to override the configured `sequence_length` of
            the layer.
        mode: The mode of operation for the layer. One of `"training"`,
            `"generate_preprocessing"` or `"generate_postprocessing'`.

    Examples:
    ```python
    # Load the preprocessor from a preset.
    preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
        "gpt2_base_en"
    )

    # Tokenize and pack a single sentence.
    sentence = tf.constant("League of legends")
    preprocessor(sentence)
    # Same output.
    preprocessor("League of legends")

    # Tokenize a batch of sentences.
    sentences = tf.constant(["Taco tuesday", "Fish taco please!"])
    preprocessor(sentences)
    # Same output.
    preprocessor(["Taco tuesday", "Fish taco please!"])

    # Map a dataset to preprocess a single sentence.
    features = tf.constant(
        [
            "Avatar 2 is amazing!",
            "Well, I am not sure.",
        ]
    )
    labels = tf.constant([1, 0])
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Map a dataset to preprocess unlabled sentences.
    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    """

    def __init__(
        self,
        tokenizer,
        sequence_length=1024,
        add_start_token=True,
        add_end_token=True,
        add_end_token_for_generate=False,
        output_strings=True,
        output_special_tokens=False,
        **kwargs,
    ):
        super().__init__(
            tokenizer,
            sequence_length=sequence_length,
            add_start_token=add_start_token,
            add_end_token=add_end_token,
            **kwargs,
        )
        self.add_end_token_for_generate = add_end_token_for_generate
        self.output_strings = output_strings
        self.output_special_tokens = output_special_tokens

    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        mode="train",
        sequence_length=None,
    ):
        if y is not None or sample_weight is not None:
            logging.warning(
                "`GPT2CausalLMPreprocessor` generates `y` and `sample_weight` "
                "based on your input data, but your data already contains `y` "
                "or `sample_weight`. Your `y` and `sample_weight` will be "
                "ignored."
            )
        sequence_length = sequence_length or self.sequence_length

        if mode == "train":
            x = convert_inputs_to_list_of_tensor_segments(x)[0]
            # Truncate with one extra token to account for the truncation below.
            token_ids, padding_mask = self.packer(
                self.tokenizer(x),
                sequence_length=sequence_length + 1,
            )
            # The last token does not have a next token, so we truncate it out.
            x = {
                "token_ids": token_ids[..., :-1],
                "padding_mask": padding_mask[..., :-1],
            }
            # Target `y` will be the next token.
            y, sample_weight = token_ids[..., 1:], padding_mask[..., 1:]
            return pack_x_y_sample_weight(x, y, sample_weight)
        elif mode == "generate_preprocess":
            x = convert_inputs_to_list_of_tensor_segments(x)[0]
            token_ids, padding_mask = self.packer(
                self.tokenizer(x),
                sequence_length=sequence_length,
                add_end_value=self.add_end_token_for_generate,
            )
            return {
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            }
        elif mode == "generate_postprocess":
            token_ids, padding_mask = x["token_ids"], x["padding_mask"]
            mask = padding_mask
            if not self.output_special_tokens:
                mask = mask & (token_ids != self.tokenizer.end_token_id)
            token_ids = tf.ragged.boolean_mask(token_ids, mask)
            outputs = self.tokenizer.detokenize(token_ids)
            if self.output_strings:
                outputs = tensor_to_string_list(outputs)
            return outputs
        else:
            raise ValueError(
                f"Unknown preprocessing mode. Received `mode={mode}`"
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "add_end_token_for_generate": self.add_end_token_for_generate,
                "output_special_tokens": self.output_special_tokens,
                "output_strings": self.output_strings,
            }
        )
        return config
