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


def convert_inputs_to_dataset(
    x=None,
    y=None,
    sample_weight=None,
    batch_size=None,
):
    if isinstance(x, tf.data.Dataset):
        if y is not None:
            raise ValueError(
                "When `x` is a `tf.data.Dataset`, please do not provide "
                f"`y`. Received: `y={y}`."
            )
        if sample_weight is not None:
            raise ValueError(
                "When `x` is a `tf.data.Dataset`, please do not provide "
                f"`sample_weight`. Received: `sample_weight={sample_weight}`."
            )
        return x

    inputs = keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
    return tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size or 32)


def apply_dataset_fn(ds, fn):
    return ds.map(fn, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


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

    def preprocess(self, *args):
        """An overridable function which preprocesses samples."""
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(args)
        x, y = self.preprocess_features(x), self.preprocess_labels(y)
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    # ========================================================================
    # Below are overrides to keras.Model methods to apply the functions above.
    # ========================================================================
    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        sample_weight=None,
        validation_data=None,
        validation_split=None,
        **kwargs,
    ):
        x = convert_inputs_to_dataset(x, y, sample_weight, batch_size)
        if self.include_preprocessing:
            x = apply_dataset_fn(x, self.preprocess)

        if validation_split:
            raise ValueError(
                "`validation_split` is not supported on this model. "
                f"Received: `validation_split={validation_split}`."
            )
        if validation_data is not None:
            val_tuple = keras.utils.unpack_x_y_sample_weight(validation_data)
            validation_data = convert_inputs_to_dataset(*val_tuple, batch_size)
            if self.include_preprocessing:
                validation_data = apply_dataset_fn(
                    validation_data, self.preprocess
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
        x = convert_inputs_to_dataset(x, y, sample_weight, batch_size)
        if self.include_preprocessing:
            x = apply_dataset_fn(x, self.preprocess)

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
        x = convert_inputs_to_dataset(x, None, None, batch_size)
        if self.include_preprocessing:
            x = apply_dataset_fn(x, self.preprocess_features)

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
