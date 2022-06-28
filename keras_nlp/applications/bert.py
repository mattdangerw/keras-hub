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
import json

import tensorflow as tf
import tensorflow_models as tfm
from tensorflow import keras

import keras_nlp

GS_FOLDER = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12/"


class BertBaseUncasedEn(keras.Model):
    """
    An uncased, English language BERT model with 110M parameters.

    This model will instantiate an architecture and pre-trained weights for the
    English language "base" variant of the BERT model. More details can be found
    in the [BERT paper](https://arxiv.org/abs/1810.04805).

    Checkpoints and encoder architecture are provided courtesy of the
    [Tensorflow Model Garden](https://github.com/tensorflow/models).

    This model contains a function called `preprocess()`, which can be called
    on any number of input sequences. Each input will be tokenized, trimmed,
    and packed into a single sequence.

    The model can optionally be configured with a classification head with
    `include_classification_head`.

    Args:
        include_classification_head: set to `True` to include a classification
            head with softmax probabilities for `num_classes` labels.
        num_classes: The number of classification labels.
        sequence_length: The desired sequence length when truncating inputs
            when calling `preprocess()`.

    Attributes:
        tokenizer: The `keras_nlp.tokenizers.WordPieceTokenizer` used to
            tokenize string inputs.
    """

    def __init__(
        self,
        checkpoint="bert",
        include_classification_head=False,
        num_classes=2,
        sequence_length=512,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.include_classification_head = include_classification_head
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        # Initialize preprocessing.
        self.tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
            vocabulary=GS_FOLDER + "vocab.txt",
        )
        self.packer = keras_nlp.layers.MultiSegmentPacker(
            sequence_length=sequence_length,
            start_value=self.tokenizer.token_to_id("[CLS]"),
            end_value=self.tokenizer.token_to_id("[SEP]"),
        )

        # Initialize encoder layers.
        encoder_config = tfm.nlp.encoders.EncoderConfig(
            type="bert",
            bert=json.load(tf.io.gfile.GFile(GS_FOLDER + "bert_config.json")),
        )
        self.encoder = tfm.nlp.encoders.build_encoder(encoder_config)
        checkpoint = tf.train.Checkpoint(encoder=self.encoder)
        checkpoint.read(GS_FOLDER + "bert_model.ckpt").assert_consumed()

        if self.include_classification_head:
            self.classification_head = keras.layers.Dense(
                num_classes,
                activation="softmax",
            )

    def preprocess(self, inputs):
        inputs = [self.tokenizer(x) for x in inputs]
        token_ids, segment_ids = self.packer(inputs)
        # Make sure the input names match the tfm encoder.
        return {
            "input_word_ids": token_ids,
            "input_type_ids": segment_ids,
            "input_mask": token_ids != 0,
        }

    def call(self, inputs):
        outputs = self.encoder(inputs)

        if self.include_classification_head:
            outputs = self.classification_head(outputs["pooled_output"])
        return outputs
