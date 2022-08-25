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

"""A base class for models including preprocessing."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.engine import data_adapter


class PipelineModel(keras.Model):
    """A model which allows automatically applying preprocessing."""

    def __init__(self, *args, include_preprocessing=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_preprocessing = include_preprocessing

    def preprocess_features(self, x):
        """An overridable function which preprocesses features."""
        return x

    def preprocess_labels(self, y):
        """An overridable function which preprocesses labels."""
        return y

    def preprocess(self, data):
        """An overridable function which preprocesses samples."""
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        x = self.preprocess_features(x)
        y = self.preprocess_labels(y)
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    # ========================================================================
    # Below are overrides to keras.Model methods to apply the functions above.
    # ========================================================================
    def train_step(self, data):
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True, include_preprocessing=False)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, sample_weight)

    def predict_step(self, data):
        x, _, _ = keras.utils.unpack_x_y_sample_weight(data)
        return self(x, training=False, include_preprocessing=False)

    def test_step(self, data):
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        y_pred = self(x, training=False, include_preprocessing=False)
        # Updates stateful loss metrics.
        self.compute_loss(x, y, y_pred, sample_weight)
        return self.compute_metrics(x, y, y_pred, sample_weight)

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        sample_weight=None,
        **kwargs,
    ):
        if validation_split and validation_data is None:
            (
                x,
                y,
                sample_weight,
            ), validation_data = data_adapter.train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )

        if validation_data:
            (
                val_x,
                val_y,
                val_sample_weight,
            ) = keras.utils.unpack_x_y_sample_weight(validation_data)

        x = data_adapter.select_data_adapter(x, y)(
            x=x,
            y=y,
            sample_weights=sample_weight,
            batch_size=batch_size,
            shuffle=shuffle,
            model=self,
        ).get_dataset()
        if self.include_preprocessing:
            x = x.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)

        validation_data = data_adapter.select_data_adapter(x, y)(
            x=val_x,
            y=val_y,
            sample_weights=val_sample_weight,
            batch_size=batch_size,
            shuffle=shuffle,
            model=self,
        ).get_dataset()
        if self.include_preprocessing:
            validation_data = validation_data.map(
                self.preprocess, num_parallel_calls=tf.data.AUTOTUNE
            )

        return super().fit(
            x=x,
            y=None,
            batch_size=None,
            sample_weight=None,
            validation_data=validation_data,
            **kwargs,
        )

    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        sample_weight=None,
        **kwargs,
    ):
        x = data_adapter.select_data_adapter(x, y)(
            x=x,
            y=y,
            sample_weights=sample_weight,
            batch_size=batch_size,
            model=self,
        ).get_dataset()
        if self.include_preprocessing:
            x = x.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)

        return super().evaluate(
            x=x,
            y=None,
            batch_size=None,
            sample_weight=None,
            **kwargs,
        )

    def predict(
        self,
        x=None,
        batch_size=None,
        **kwargs,
    ):
        x = data_adapter.select_data_adapter(x, None)(
            x=x,
            y=None,
            batch_size=batch_size,
            model=self,
        ).get_dataset()
        if self.include_preprocessing:
            x = x.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)

        return super().predict(
            x=x,
            batch_size=None,
            **kwargs,
        )

    def __call__(self, inputs, include_preprocessing=None, **kwargs):
        if include_preprocessing is None:
            include_preprocessing = self.include_preprocessing
        if include_preprocessing:
            inputs = self.preprocess_features(inputs)
        return super().__call__(inputs, **kwargs)
