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
"""Tests for FNet classification model."""

import io
import os

import pytest
import sentencepiece
import tensorflow as tf
from absl.testing import parameterized

from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.models.f_net.f_net_backbone import FNetBackbone
from keras_nlp.models.f_net.f_net_classifier import FNetClassifier
from keras_nlp.models.f_net.f_net_preprocessor import FNetPreprocessor
from keras_nlp.models.f_net.f_net_tokenizer import FNetTokenizer
from keras_nlp.tests.test_case import TestCase


@pytest.mark.tf_only
class FNetClassifierTest(TestCase):
    def setUp(self):
        # Setup Model
        bytes_io = io.BytesIO()
        vocab_data = tf.data.Dataset.from_tensor_slices(
            ["the quick brown fox", "the earth is round"]
        )

        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=vocab_data.as_numpy_iterator(),
            model_writer=bytes_io,
            vocab_size=12,
            model_type="WORD",
            pad_id=3,
            unk_id=0,
            bos_id=4,
            eos_id=5,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="[CLS]",
            eos_piece="[SEP]",
            user_defined_symbols="[MASK]",
        )

        self.proto = bytes_io.getvalue()

        self.preprocessor = FNetPreprocessor(
            tokenizer=FNetTokenizer(proto=self.proto),
            sequence_length=8,
        )
        self.backbone = FNetBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            hidden_dim=2,
            intermediate_dim=4,
            max_sequence_length=self.preprocessor.packer.sequence_length,
        )
        self.classifier = FNetClassifier(
            self.backbone,
            num_classes=4,
            preprocessor=self.preprocessor,
            # Check we handle serialization correctly.
            activation=keras.activations.softmax,
        )

        # Setup data.
        self.raw_batch = [
            "the quick brown fox.",
            "the slow brown fox.",
        ]
        self.preprocessed_batch = self.preprocessor(self.raw_batch)
        self.raw_dataset = tf.data.Dataset.from_tensor_slices(
            (self.raw_batch, ops.ones((2,)))
        ).batch(2)
        self.preprocessed_dataset = self.raw_dataset.map(self.preprocessor)

    def test_valid_call_classifier(self):
        self.classifier(self.preprocessed_batch)

    def test_classifier_predict(self):
        preds1 = self.classifier.predict(self.raw_batch)
        self.classifier.preprocessor = None
        preds2 = self.classifier.predict(self.preprocessed_batch)
        # Assert predictions match.
        self.assertAllClose(preds1, preds2)
        # Assert valid softmax output.
        self.assertAllClose(ops.sum(preds2, axis=-1), [1.0, 1.0])

    def test_fnet_classifier_fit(self):
        self.classifier.fit(self.raw_dataset)
        self.classifier.preprocessor = None
        self.classifier.fit(self.preprocessed_dataset)

    def test_classifier_fit_no_xla(self):
        self.classifier.preprocessor = None
        self.classifier.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            jit_compile=False,
        )
        self.classifier.fit(self.preprocessed_dataset)

    def test_serialization(self):
        # Defaults.
        original = FNetClassifier(
            self.backbone,
            num_classes=2,
        )
        config = keras.saving.serialize_keras_object(original)
        restored = keras.saving.deserialize_keras_object(config)
        self.assertEqual(restored.get_config(), original.get_config())
        # With options.
        original = FNetClassifier(
            self.backbone,
            num_classes=4,
            preprocessor=self.preprocessor,
            activation=keras.activations.softmax,
            name="test",
            trainable=False,
        )
        config = keras.saving.serialize_keras_object(original)
        restored = keras.saving.deserialize_keras_object(config)
        self.assertEqual(restored.get_config(), original.get_config())

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    @pytest.mark.large
    def test_saved_model(self, save_format, filename):
        model_output = self.classifier.predict(self.raw_batch)
        path = os.path.join(self.get_temp_dir(), filename)
        # Don't save traces in the tf format, we check compilation elsewhere.
        kwargs = {"save_traces": False} if save_format == "tf" else {}
        self.classifier.save(path, save_format=save_format, **kwargs)
        restored_model = keras.models.load_model(path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, FNetClassifier)

        # Check that output matches.
        restored_output = restored_model.predict(self.raw_batch)
        self.assertAllClose(model_output, restored_output)
