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

"""Bert model configurable class, preconfigured versions, and task heads."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers.multi_segment_packer import MultiSegmentPacker
from keras_nlp.layers.position_embedding import PositionEmbedding
from keras_nlp.layers.transformer_encoder import TransformerEncoder
from keras_nlp.tokenizers.word_piece_tokenizer import WordPieceTokenizer
from keras_nlp.utils.pipeline_model import PipelineModel


def _bert_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


# Pretrained models
BASE_PATH = "https://storage.googleapis.com/keras-nlp/models/"

checkpoints = {
    "bert_base": {
        "uncased_en": {
            "md5": "9b2b2139f221988759ac9cdd17050b31",
            "description": "Base size of Bert where all input is lowercased. "
            "Trained on English wikipedia + books corpora.",
            "vocabulary_size": 30522,
        },
        "cased_en": {
            "md5": "f94a6cb012e18f4fb8ec92abb91864e9",
            "description": "Base size of Bert where case is maintained. "
            "Trained on English wikipedia + books corpora.",
            "vocabulary_size": 28996,
        },
    }
}


class BertCustomNetwork(keras.Model):
    """Bi-directional Transformer-based encoder network.

    This network implements a bi-directional Transformer-based encoder as
    described in ["BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding"](https://arxiv.org/abs/1810.04805). It includes the
    embedding lookups and transformer layers, but not the masked language model
    or classification task networks.

    This class gives a fully customizable Bert model with any number of layers,
    heads, and embedding dimensions. For specific specific bert architectures
    defined in the paper, see for example `keras_nlp.models.BertNetwork`.

    Args:
        vocabulary_size: Int. The size of the token vocabulary.
        num_layers: Int. The number of transformer layers.
        num_heads: Int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: Int. The size of the transformer encoding and pooler layers.
        intermediate_dim: Int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        dropout: Float. Dropout probability for the Transformer encoder.
        max_sequence_length: Int. The maximum sequence length that this encoder
            can consume. If None, `max_sequence_length` uses the value from
            sequence length. This determines the variable shape for positional
            embeddings.
        num_segments: Int. The number of types that the 'segment_ids' input can
            take.
        name: String, optional. Name of the model.
        trainable: Boolean, optional. If the model's variables should be
            trainable.

    Example usage:
    ```python
    # Randomly initialized Bert encoder
    model = keras_nlp.models.BertCustomNetwork(
        vocabulary_size=30522,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12,
        name="encoder",
    )

    # Call encoder on the inputs
    input_data = {
        "input_ids": tf.random.uniform(
            shape=(1, 12), dtype=tf.int64, maxval=model.vocabulary_size
        ),
        "segment_ids": tf.constant(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
        "input_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
    }
    output = model(input_data)
    ```
    """

    # TODO(jbischof): consider changing `intermediate_dim` and `hidden_dim` to
    # less confusing name here and in TransformerEncoder (`feed_forward_dim`?)

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout=0.1,
        max_sequence_length=512,
        num_segments=2,
        name=None,
        trainable=True,
    ):

        # Index of classification token in the vocabulary
        cls_token_index = 0
        # Inputs
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="input_ids"
        )
        segment_id_input = keras.Input(
            shape=(None,), dtype="int32", name="segment_ids"
        )
        input_mask = keras.Input(
            shape=(None,), dtype="int32", name="input_mask"
        )

        # Embed tokens, positions, and segment ids.
        token_embedding_layer = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=_bert_kernel_initializer(),
            name="token_embedding",
        )
        token_embedding = token_embedding_layer(token_id_input)
        position_embedding = PositionEmbedding(
            initializer=_bert_kernel_initializer(),
            sequence_length=max_sequence_length,
            name="position_embedding",
        )(token_embedding)
        segment_embedding = keras.layers.Embedding(
            input_dim=num_segments,
            output_dim=hidden_dim,
            embeddings_initializer=_bert_kernel_initializer(),
            name="segment_embedding",
        )(segment_id_input)

        # Sum, normailze and apply dropout to embeddings.
        x = keras.layers.Add()(
            (token_embedding, position_embedding, segment_embedding)
        )
        x = keras.layers.LayerNormalization(
            name="embeddings_layer_norm",
            axis=-1,
            epsilon=1e-12,
            dtype=tf.float32,
        )(x)
        x = keras.layers.Dropout(
            dropout,
            name="embeddings_dropout",
        )(x)

        # Apply successive transformer encoder blocks.
        for i in range(num_layers):
            x = TransformerEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                activation=lambda x: keras.activations.gelu(
                    x, approximate=True
                ),
                dropout=dropout,
                kernel_initializer=_bert_kernel_initializer(),
                name=f"transformer_layer_{i}",
            )(x, padding_mask=input_mask)

        # Construct the two Bert outputs. The pooled output is a dense layer on
        # top of the [CLS] token.
        sequence_output = x
        pooled_output = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=_bert_kernel_initializer(),
            activation="tanh",
            name="pooled_dense",
        )(x[:, cls_token_index, :])

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs={
                "input_ids": token_id_input,
                "segment_ids": segment_id_input,
                "input_mask": input_mask,
            },
            outputs={
                "sequence_output": sequence_output,
                "pooled_output": pooled_output,
            },
            name=name,
            trainable=trainable,
        )
        # All references to `self` below this line
        self.token_embedding = token_embedding_layer
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_sequence_length = max_sequence_length
        self.num_segments = num_segments
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.cls_token_index = cls_token_index

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "max_sequence_length": self.max_sequence_length,
                "num_segments": self.num_segments,
                "dropout": self.dropout,
                "cls_token_index": self.cls_token_index,
            }
        )
        return config


def BertNetwork(
    architecture="base",
    weights="uncased_en",
    vocabulary_size=None,
    name=None,
    trainable=True,
):
    """Bert

    This network implements a bi-directional Transformer-based encoder as
    described in ["BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding"](https://arxiv.org/abs/1810.04805). It includes the
    embedding lookups and transformer layers, but not the masked language model
    or classification task networks.

    Args:
        weights: String, optional. Name of pretrained model to load weights.
            If None, model is randomly initialized. Either `weights` or
            `vocabulary_size` must be specified, but not both.
        vocabulary_size: Int, optional. The size of the token vocabulary. Either
            `weights` or `vocabularly_size` must be specified, but not both.
        name: String, optional. Name of the model.
        trainable: Boolean, optional. If the model's variables should be
            trainable.

    Example usage:
    ```python
    # Randomly initialized BertNetwork encoder
    model = keras_nlp.models.BertNetwork(vocabulary_size=10000)

    # Call encoder on the inputs.
    input_data = {{
        "input_ids": tf.random.uniform(
            shape=(1, 512), dtype=tf.int64, maxval=model.vocabulary_size
        ),
        "segment_ids": tf.constant([0] * 200 + [1] * 312, shape=(1, 512)),
        "input_mask": tf.constant([1] * 512, shape=(1, 512)),
    }}
    output = model(input_data)

    # Load a pretrained model
    model = keras_nlp.models.BertNetwork(weights="uncased_en")
    # Call encoder on the inputs.
    output = model(input_data)
    ```
    """
    if architecture != "base":
        raise ValueError("Only `architecture='base'` is currently supported.")

    if weights:
        if weights not in checkpoints["bert_base"]:
            raise ValueError(
                "`weights` must be one of "
                f"""{", ".join(checkpoints["bert_base"])}. """
                f"Received: {weights}"
            )
        vocabulary_size = checkpoints["bert_base"][weights]["vocabulary_size"]

    model = BertCustomNetwork(
        vocabulary_size=vocabulary_size,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        dropout=0.1,
        max_sequence_length=512,
        name=name,
        trainable=trainable,
    )

    # TODO(jbischof): consider changing format from `h5` to
    # `tf.train.Checkpoint` once
    # https://github.com/keras-team/keras/issues/16946 is resolved
    if weights:
        filepath = keras.utils.get_file(
            "model.h5",
            BASE_PATH + "bert_base_" + weights + "/model.h5",
            cache_subdir="models/bert_base/" + weights + "/",
            file_hash=checkpoints["bert_base"][weights]["md5"],
        )
        model.load_weights(filepath)

    return model


class BertTokenizer(WordPieceTokenizer):
    """WordPieceTokenizer with pretrained Bert vocabularies.

    This class is a specialized instance of
    `keras_nlp.tokenizers.WordPieceTokenizer` which supports loading pre-trained
    vocabularies to match pre-trained Bert models.

    Args:
        vocabulary: The name of the pre-trained vocabulary to load.
    """

    def __init__(
        self,
        vocabulary,
    ):
        if vocabulary:
            vocabulary = keras.utils.get_file(
                "vocab.txt",
                BASE_PATH + "bert_base_" + vocabulary + "/vocab.txt",
                cache_subdir="models/bert_base/" + vocabulary + "/",
            )
        super().__init__(vocabulary=vocabulary)

        # Convenience accessors for special tokens.
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        self.pad_token_id = self.token_to_id(self.pad_token)
        self.cls_token_id = self.token_to_id(self.cls_token)
        self.sep_token_id = self.token_to_id(self.sep_token)
        self.mask_token_id = self.token_to_id(self.mask_token)


class BertClassificationHead(keras.layers.Layer):
    """Bert classification head.

    This layer should be called directly on the outputs of a
    `keras_nlp.models.BertNetwork`, an will produce a output with shape


    Args:
        num_classes: Int. Number of classes to predict.
        activation: The activation of the dense output layer. Usually this
            should be set to either `None` for class logits, or "softmax" for
            class probabilities.
        name: String, optional. Name of the layer.
    """

    def __init__(
        self,
        num_classes,
        activation="softmax",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.activation = keras.activations.get(activation)

        self.dense = keras.layers.Dense(
            num_classes,
            kernel_initializer=_bert_kernel_initializer(),
            activation=self.activation,
            name="logits",
        )

    def call(self, inputs):
        return self.dense(inputs["pooled_output"])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "activation": keras.activations.serialize(self.activation),
            }
        )
        return config


class BertClassifier(PipelineModel):
    """Bert classification workflow.

    This is a model which has preprocessing build in and is precompiled to
    support easy Bert finetuning for classification tasks.

    Args:
        num_classes: Int. Number of classes to predict.
        weights: The pretrained weight to load.
        architecture: One of "base" or "large".
        name: String, optional. Name of the model.
        trainable: Boolean, optional. If the model's variables should be
            trainable.
    ```
    """

    def __init__(
        self,
        num_classes,
        architecture="base",
        weights="uncased_en",
        sequence_length=512,
        name=None,
    ):
        # Workflow components.
        tokenizer = BertTokenizer(
            vocabulary=weights,
        )
        packer = MultiSegmentPacker(
            sequence_length=sequence_length,
            start_value=tokenizer.cls_token_id,
            end_value=tokenizer.sep_token_id,
        )
        network = BertNetwork(
            architecture=architecture,
            weights=weights,
        )
        head = BertClassificationHead(
            num_classes=num_classes,
            activation="softmax",
        )

        super().__init__(
            inputs=network.input,
            outputs=head(network(network.input)),
            name=name,
        )

        # Config arguments.
        self.architecture = architecture
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        # Component layers.
        self.tokenizer = tokenizer
        self.packer = packer
        self.network = network
        self.head = head

        # Compile with reasonable defaults.
        self.compile(
            optimizer=keras.optimizers.Adam(learning_rate=2e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
            jit_compile=True,
        )

    # TODO(mattdangerw): figure out how to remove this.
    def is_preprocessed(self, x):
        return isinstance(x, dict)

    def preprocess_features(self, x):
        if self.is_preprocessed(x):
            return x
        if isinstance(x, str):
            x = tf.convert_to_tensor(x)
        if not isinstance(x, (list, tuple)):
            x = [x]
        input_ids, segment_ids = self.packer([self.tokenizer(s) for s in x])
        if input_ids.shape.rank == 1:
            input_ids = tf.expand_dims(input_ids, 0)
            segment_ids = tf.expand_dims(segment_ids, 0)
        return {
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "input_mask": input_ids != 0,
        }
