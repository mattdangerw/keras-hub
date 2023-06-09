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
import math

from tensorflow import keras

from keras_nlp.layers.transformer_encoder import TransformerEncoder


class LoraBackbone(keras.Model):
    def __init__(
        self,
        original_backbone,
        **kwargs,
    ):
        inputs = self.input
        token_embedding = self.get_layer("token_embedding")(inputs["token_ids"])
        position_embedding = self.get_layer("position_embedding")(
            token_embedding
        )
        x = self.get_layer("embeddings_add")(
            (token_embedding, position_embedding)
        )
        x = self.get_layer("embeddings_layer_norm")(x)
        x = self.get_layer("embeddings_dropout")(x)
        for i in range(self.num_layers):
            layer = self.get_layer(f"transformer_layer_{i}")
            lora_layer = LoraTransformerEncoder(layer)
            x = lora_layer(f"transformer_layer_{i}")(
                x, padding_mask=inputs["padding_mask"]
            )
        sequence_output = x
        pooled_output = self.get_layer("pooled_dense")(x)
        super().__init__(
            inputs=inputs,
            outputs={
                "sequence_output": sequence_output,
                "pooled_output": pooled_output,
            },
        )

    @property
    def token_embedding(self):
        raise self.original_backbone.token_embedding


class LoraTransformerEncoder(TransformerEncoder):
    def _build(self, input_shape):
        super()._build(input_shape)
        self._self_attention_layer = LoraMultiHeadAttention(
            self._self_attention_layer
        )


class LoraMultiHeadAttention(keras.layers.Layer):
    def __init__(
        self,
        original_layer,
        rank=8,
        alpha=32,
        **kwargs,
    ):
        # We want to keep the name of this layer the same as the original
        # dense layer.
        original_config = original_layer.get_config()
        name = original_config["name"]

        kwargs.pop("name", None)

        super().__init__(name=name, **kwargs)

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Original dense layer.
        self.original_layer = original_layer

        # LoRA dense layers.
        self.A = keras.layers.Dense(
            units=rank,
            use_bias=False,
            kernel_initializer=keras.initializers.VarianceScaling(
                scale=math.sqrt(5), mode="fan_in", distribution="uniform"
            ),
            name=f"lora_A",
        )
        self.B = keras.layers.EinsumDense(
            equation=original_config["equation"],
            output_shape=original_config["output_shape"],
            kernel_initializer="zeros",
            name=f"lora_B",
        )

    def call(self, inputs):
        original_output = self.original_layer(inputs)
        lora_output = self.B(self.A(inputs)) * self.scale
        return original_output + lora_output
