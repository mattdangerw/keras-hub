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

"""T5 backbone model."""

import copy

import tensorflow as tf
from tensorflow import keras

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.layers.transformer_layer_utils import compute_causal_mask
from keras_nlp.models.backbone import Backbone
from keras_nlp.models.t5.t5_layer_norm import T5LayerNorm
from keras_nlp.models.t5.t5_multi_head_attention import T5MultiHeadAttention
from keras_nlp.models.t5.t5_presets import backbone_presets
from keras_nlp.utils.python_utils import classproperty


@keras_nlp_export("keras_nlp.models.T5Backbone")
class T5Backbone(Backbone):
    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        activation="relu",
        dropout=0.1,
        layer_norm_epsilon=1e-06,
        **kwargs,
    ):
        # Encoder inputs
        encoder_token_ids = keras.Input(
            shape=(None,), dtype="int32", name="encoder_token_ids"
        )
        encoder_padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="encoder_padding_mask"
        )

        # Decoder inputs.
        decoder_token_ids = keras.Input(
            shape=(None,), dtype="int32", name="decoder_token_ids"
        )
        decoder_padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="decoder_padding_mask"
        )

        # Token embedding layer. This layer is shared by encoder and decoder.
        token_embedding_layer = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(1.0),
            name="token_embedding",
        )

        # ===== Encoder =====

        # Embed tokens and positions.
        token_embedding = token_embedding_layer(encoder_token_ids)
        x = keras.layers.Dropout(dropout)(token_embedding)

        position_bias = None
        for i in range(num_layers):
            layer_outputs = T5Encoder(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                activation=activation,
                layer_norm_epsilon=layer_norm_epsilon,
                num_heads=num_heads,
                use_relative_attention_bias=bool(i == 0),
                name=f"transformer_encoder_layer_{i}",
            )(
                x,
                padding_mask=encoder_padding_mask,
                position_bias=position_bias,
            )
            x = layer_outputs["hidden_states"]
            position_bias = layer_outputs["position_bias"]

        x = keras.layers.Dropout(
            dropout,
            name="encoder_output_dropout",
        )(x)
        x = T5LayerNorm(
            epsilon=layer_norm_epsilon,
            name="encoder_output_layer_norm",
        )(x)

        encoder_output = x

        # ===== Decoder =====

        # Embed tokens and positions.
        token_embedding = token_embedding_layer(decoder_token_ids)
        x = keras.layers.Dropout(dropout)(token_embedding)

        position_bias = None
        cross_attention_position_bias = None
        for i in range(num_layers):
            layer_outputs = T5Decoder(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                activation=activation,
                layer_norm_epsilon=layer_norm_epsilon,
                num_heads=num_heads,
                use_relative_attention_bias=bool(i == 0),
                name=f"transformer_decoder_layer_{i}",
            )(
                x,
                encoder_output,
                encoder_padding_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                position_bias=position_bias,
                cross_attention_position_bias=cross_attention_position_bias,
            )
            x = layer_outputs["hidden_states"]
            position_bias = layer_outputs["position_bias"]
            cross_attention_position_bias = layer_outputs[
                "cross_attention_position_bias"
            ]

        x = keras.layers.Dropout(
            dropout,
            name="decoder_output_dropout",
        )(x)
        x = T5LayerNorm(
            epsilon=layer_norm_epsilon,
            name="decoder_output_layer_norm",
        )(x)

        decoder_output = x

        super().__init__(
            inputs={
                "encoder_token_ids": encoder_token_ids,
                "encoder_padding_mask": encoder_padding_mask,
                "decoder_token_ids": decoder_token_ids,
                "decoder_padding_mask": decoder_padding_mask,
            },
            outputs={
                "encoder_sequence_output": encoder_output,
                "decoder_sequence_output": decoder_output,
            },
            **kwargs,
        )

        # All references to `self` below this line
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "activation": self.activation,
                "dropout": self.dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    @property
    def token_embedding(self):
        return self.get_layer("token_embedding")

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)


class T5DenseBlock(keras.layers.Layer):
    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        dropout,
        activation,
        layer_norm_epsilon,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_projector = keras.layers.Dense(
            intermediate_dim,
            use_bias=False,
            name="input_projector",
            activation=keras.activations.get(activation),
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=hidden_dim**-0.5
            ),
        )
        self.output_projector = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            name="output_projector",
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=intermediate_dim**-0.5
            ),
        )
        self.layer_norm = T5LayerNorm(epsilon=layer_norm_epsilon)
        self.dropout_layer = keras.layers.Dropout(dropout)

    def call(self, inputs, training=False):
        hidden_states = self.layer_norm(inputs)
        hidden_states = self.input_projector(hidden_states)
        hidden_states = self.dropout_layer(hidden_states, training=training)
        hidden_states = self.output_projector(hidden_states)
        return inputs + self.dropout_layer(hidden_states, training=training)


class T5AttentionBlock(keras.layers.Layer):
    def __init__(
        self,
        is_decoder,
        hidden_dim,
        num_heads,
        dropout,
        layer_norm_epsilon,
        use_relative_attention_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attention = T5MultiHeadAttention(
            is_decoder,
            hidden_dim,
            num_heads,
            dropout,
            use_relative_attention_bias=use_relative_attention_bias,
        )
        self.layer_norm = T5LayerNorm(epsilon=layer_norm_epsilon)
        self.dropout_layer = keras.layers.Dropout(dropout)

    def call(
        self,
        hidden_states,
        key_value_states=None,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        query_length=None,
        training=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output, position_bias = self.attention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            query_length=query_length,
            training=training,
        )
        hidden_states = hidden_states + self.dropout_layer(
            attention_output, training=training
        )
        return (hidden_states, position_bias)


class T5Encoder(keras.layers.Layer):
    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        dropout,
        activation,
        layer_norm_epsilon,
        num_heads,
        use_relative_attention_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.self_attention = T5AttentionBlock(
            is_decoder=False,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            layer_norm_epsilon=layer_norm_epsilon,
            use_relative_attention_bias=use_relative_attention_bias,
        )
        self.dense_block = T5DenseBlock(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
            activation=activation,
            layer_norm_epsilon=layer_norm_epsilon,
        )

    def call(
        self,
        hidden_states,
        padding_mask=None,
        position_bias=None,
        training=False,
    ):
        attention_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        hidden_states, position_bias = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            training=training,
        )

        hidden_states = self.dense_block(hidden_states, training=training)
        return {
            "hidden_states": hidden_states,
            "position_bias": position_bias,
        }


class T5Decoder(keras.layers.Layer):
    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        dropout,
        activation,
        layer_norm_epsilon,
        num_heads,
        use_relative_attention_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.self_attention = T5AttentionBlock(
            is_decoder=True,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            layer_norm_epsilon=layer_norm_epsilon,
            use_relative_attention_bias=use_relative_attention_bias,
        )
        self.cross_attention = T5AttentionBlock(
            is_decoder=True,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            layer_norm_epsilon=layer_norm_epsilon,
        )
        self.dense_block = T5DenseBlock(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
            activation=activation,
            layer_norm_epsilon=layer_norm_epsilon,
        )

    def call(
        self,
        decoder_hidden_states,
        encoder_hidden_states,
        decoder_padding_mask=None,
        encoder_padding_mask=None,
        position_bias=None,
        cross_attention_position_bias=None,
        training=False,
    ):
        encoder_mask = encoder_padding_mask[:, tf.newaxis, tf.newaxis, :]
        decoder_mask = decoder_padding_mask[:, tf.newaxis, tf.newaxis, :]
        # Add a causal mask for the decoder.
        batch_size = tf.shape(decoder_hidden_states)[0]
        length = tf.shape(decoder_hidden_states)[1]
        causal_mask = compute_causal_mask(batch_size, length, length)
        decoder_mask = decoder_mask & causal_mask[:, tf.newaxis, :, :]

        hidden_states, position_bias = self.self_attention(
            decoder_hidden_states,
            attention_mask=decoder_mask,
            position_bias=position_bias,
            training=training,
        )
        hidden_states, cross_attention_position_bias = self.cross_attention(
            decoder_hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_mask,
            position_bias=cross_attention_position_bias,
            query_length=None,
            training=training,
        )
        hidden_states = self.dense_block(hidden_states, training=training)
        return {
            "hidden_states": hidden_states,
            "position_bias": position_bias,
            "cross_attention_position_bias": cross_attention_position_bias,
        }
