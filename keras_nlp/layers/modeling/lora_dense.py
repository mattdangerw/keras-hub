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

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.backend import keras
from keras_nlp.backend import ops


@keras_nlp_export("keras_nlp.layers.LoraDense")
class LoraDense(keras.layers.Layer):
    """A LoRA adapter layer for a dense input layer."""

    def __init__(
        self,
        inner_dense,
        rank=8,
        alpha=32,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.inner_dense = inner_dense

        self.lora_a = keras.layers.Dense(
            units=rank,
            use_bias=False,
            # Note: the original paper mentions that normal distribution was
            # used for initialization. However, the official LoRA implementation
            # uses "Kaiming/He Initialization".
            kernel_initializer=keras.initializers.VarianceScaling(
                scale=np.sqrt(5), mode="fan_in", distribution="uniform"
            ),
            name="lora_a",
        )
        if isinstance(inner_dense, keras.layers.EinsumDense):
            self.lora_b = keras.layers.EinsumDense(
                equation=inner_dense.equation,
                output_shape=inner_dense.partial_output_shape,
                kernel_initializer="zeros",
                name="lora_b",
            )
        elif isinstance(inner_dense, keras.layers.Dense):
            self.lora_b = keras.layers.Dense(
                units=inner_dense.units,
                kernel_initializer="zeros",
                use_bias=False,
                name="lora_b",
            )
        else:
            raise ValueError(
                "Only `Dense` and `EinsumDense` inner layers are supported. "
                f"Received: layer={inner_dense}"
            )

        if inner_dense.built:
            self.build_from_config(inner_dense.get_build_config())

    def build(self, inputs_shape):
        self.lora_a.build(inputs_shape)
        self.lora_b.build(self.lora_a.compute_output_shape(inputs_shape))
        self.built = True

    def merge_weights(self):
        """Merge lora updates into the wrapped dense layer.

        This function should only be called outside of any compiled context
        (e.g. not during `fit()`, `predict()` or `evaluate()`). It will merge
        the updates from the lora layers into the original dense layer, and
        re-initialize the lor
        """
        # Add lora updates back into the inner dense kernel.
        updates = ops.squeeze(
            self.lora_b(ops.expand_dims(self.lora_a.kernel, 0)), 0
        )
        self.inner_dense.kernel.assign_add(updates)
        # Re-initialize lora weights.
        self.lora_a.kernel.assign(
            self.lora_a.kernel_initializer(
                self.lora_a.kernel.shape, self.lora_a.kernel.dtype
            )
        )
        self.lora_b.kernel.assign(
            self.lora_b.kernel_initializer(
                self.lora_b.kernel.shape, self.lora_b.kernel.dtype
            )
        )

    def call(self, inputs):
        original_output = self.inner_dense(inputs)
        # If we are fine-tuning the model, we will add LoRA layers' output
        # to the original layer's output.
        lora_output = self.lora_b(self.lora_a(inputs)) * self.scale
        return original_output + lora_output

    @classmethod
    def from_config(cls, config):
        config["inner_dense"] = keras.deserialize_keras_object(
            config["inner_dense"]
        )
        return super().from_config(config)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "inner_dense": self.serialize_keras_object(self.inner_dense),
            }
        )
        return config
