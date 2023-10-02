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

import copy

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.backend import keras
from keras_nlp.layers.modeling.token_and_position_embedding import (
    TokenAndPositionEmbedding,
)
from keras_nlp.layers.modeling.transformer_decoder import TransformerDecoder
from keras_nlp.models.backbone import Backbone
from keras_nlp.models.opt.opt_presets import backbone_presets
from keras_nlp.utils.python_utils import classproperty


def opt_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


@keras_nlp_export("keras_nlp.models.OPTBackbone")
class OPTBackbone(Backbone):
    """An OPT decoder network.

    This class implements a Transformer-based decoder model as described in
    ["OPT: Open Pre-trained Transformer Language Models"](https://arxiv.org/abs/2205.01068).
    The default constructor gives a fully customizable, randomly initialized OPT
    model with any number of layers, heads, and embedding dimensions. To load
    preset architectures and weights, use the `from_preset()` constructor.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/facebookresearch/fairseq/).

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer decoder layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The hidden size of the transformer decoder layers.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer decoder layer.
        dropout: float. Dropout probability for the Transformer decoder.
        max_sequence_length: int. The maximum sequence length that this decoder
            can consume. If `None`, `max_sequence_length` uses the value from
            sequence length. This determines the variable shape for positional
            embeddings.

    Examples:
    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Pretrained OPT decoder
    model = keras_nlp.models.OPTBackbone.from_preset("opt_125m_en")
    model(input_data)

    # Randomly initialized OPT decoder model with a custom config
    model = keras_nlp.models.OPTBackbone(
        vocabulary_size=50265,
        num_layers=4,
        num_heads=4,
        hidden_dim=256,
        intermediate_dim=512,
        max_sequence_length=128,
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout=0.1,
        max_sequence_length=2048,
        **kwargs,
    ):
        # Decoder inputs.
        token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Embed tokens and positions.
        embedding_layer = TokenAndPositionEmbedding(
            vocabulary_size=vocabulary_size,
            sequence_length=max_sequence_length,
            embedding_dim=hidden_dim,
            embeddings_initializer=opt_kernel_initializer(),
            name="embeddings",
        )
        x = embedding_layer(token_ids)

        # Apply successive transformer decoder blocks.
        for i in range(num_layers):
            x = TransformerDecoder(
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation="relu",
                layer_norm_epsilon=1e-5,
                normalize_first=True,
                kernel_initializer=opt_kernel_initializer(),
                name=f"transformer_layer_{i}",
            )(x, decoder_padding_mask=padding_mask)

        # Add a final layer norm.
        x = keras.layers.LayerNormalization(
            name="layer_norm",
            axis=-1,
            epsilon=1e-5,
            dtype="float32",
        )(x)

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs={
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            },
            outputs=x,
            **kwargs,
        )

        # All references to `self` below this line
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        self.token_embedding = embedding_layer.token_embedding

    def get_config(self):
        return {
            "vocabulary_size": self.vocabulary_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": self.intermediate_dim,
            "dropout": self.dropout,
            "max_sequence_length": self.max_sequence_length,
        }

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classmethod
    def create_layout_map(cls, device_mesh):
        """Create a layout map for model parallel training.

        This method takes in a `keras.distribution.DeviceMesh` and returns a
        `keras.distribution.LayoutMap` that will correctly distribute weights
        for a `keras_nlp.models.GPT2Backbone in a model parallel setting.

        Args:
            device_mesh: A 2D `keras.distribution.DeviceMesh` describing the
                arrangement of devices for running distributed computation. The
                first dimension in the mesh is expected to be for data parallel
                distribution, and the second for model parallel distribution.

        Returns:
            A `tf.keras.dtensor.experimental.LayoutMap` which contains the
            proper layout to weights mapping for the model parallel setting.

        Examples:
        ```python
        device_mesh = keras.distribution.DeviceMesh(
            shape=(2, 4),
            axis_names=('batch', 'model'),
            devices=keras.distribution.list_devices(),
        )
        layout_map = keras_nlp.models.GPT2Backbone.create_layout_map(
            device_mesh,
        )
        distribution = keras.distribution.ModelParallel(device_mesh, layout_map)
        keras.distribution.set_distribution(distribution)
        ```
        """
        # We assert the mesh is 2D, and assume the first mesh dim is for data
        # parallel and the second dim is for model parallel.
        mesh_shape = device_mesh.shape
        if len(mesh_shape) != 2:
            raise ValueError(f"Expect a 2D DeviceMesh, received {device_mesh}")
        _, model_dim = device_mesh.axis_names

        layout_map = keras.distribution.LayoutMap(device_mesh=device_mesh)
        # Embedding sharding
        layout_map[r".*embeddings"] = [None, model_dim]
        # Transformer block sharding
        layout_map[r".*(query|key|value)/kernel"] = [None, None, model_dim]
        layout_map[r".*(query|key|value)/bias"] = [model_dim, None]
        layout_map[r".*feedforward_intermediate_dense/kernel"] = [None, model_dim]
        layout_map[r".*feedforward_intermediate_dense/bias"] = [model_dim]
        layout_map[r".*feedforward_output_dense/kernel"] = [model_dim, None]
        layout_map[r".*feedforward_output_dense/bias"] = [None]
        return layout_map
