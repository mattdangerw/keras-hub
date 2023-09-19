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

import numpy as np
import pytest
import tensorflow as tf

from keras_nlp.models.bert.bert_backbone import BertBackbone
from keras_nlp.tests.test_case import TestCase


class BertBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 2,
            "intermediate_dim": 4,
            "max_sequence_length": 5,
        }
        self.input_data = {
            "token_ids": np.ones((2, 5), dtype="int32"),
            "segment_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            backbone_cls=BertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "sequence_output": (2, 5, 2),
                "pooled_output": (2, 2),
            },
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            model_cls=BertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )


@pytest.mark.tpu
@pytest.mark.usefixtures("tpu_test_class")
class BertBackboneTPUTest(TestCase):
    def setUp(self):
        with self.tpu_strategy.scope():
            self.backbone = BertBackbone(
                vocabulary_size=1000,
                num_layers=2,
                num_heads=2,
                hidden_dim=64,
                intermediate_dim=128,
                max_sequence_length=128,
            )
        self.input_batch = {
            "token_ids": np.ones((8, 128), dtype="int32"),
            "segment_ids": np.ones((8, 128), dtype="int32"),
            "padding_mask": np.ones((8, 128), dtype="int32"),
        }
        self.input_dataset = tf.data.Dataset.from_tensor_slices(
            self.input_batch
        ).batch(2)

    def test_predict(self):
        self.backbone.compile()
        self.backbone.predict(self.input_dataset)
