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
            "vocabulary": "https://storage.googleapis.com/cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12/vocab.txt",
        },
        "cased_en": {
            "md5": "f94a6cb012e18f4fb8ec92abb91864e9",
            "description": "Base size of Bert where case is maintained. "
            "Trained on English wikipedia + books corpora.",
            "vocabulary": None,
        },
    }
}


class BertCustom(PipelineModel):
    """Bi-directional Transformer-based encoder network.

    This network implements a bi-directional Transformer-based encoder as
    described in ["BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding"](https://arxiv.org/abs/1810.04805). It includes the
    embedding lookups and transformer layers, but not the masked language model
    or classification task networks.

    This class gives a fully customizable Bert model with any number of layers,
    heads, and embedding dimensions. For specific specific bert architectures
    defined in the paper, see for example `keras_nlp.models.BertBase`.

    Args:
        vocabulary: A list or filename.
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
    model = keras_nlp.models.BertCustom(
        vocabulary="vocab.txt",
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
            shape=(1, 12), dtype=tf.int64, maxval=30522
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
        vocabulary,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout=0.1,
        max_sequence_length=512,
        num_segments=2,
        name=None,
        trainable=True,
        include_preprocessing=True,
    ):
        # Setup preprocessing.
        tokenizer = WordPieceTokenizer(
            vocabulary=vocabulary,
        )
        packer = MultiSegmentPacker(
            sequence_length=128,
            start_value=tokenizer.token_to_id("[CLS]"),
            end_value=tokenizer.token_to_id("[SEP]"),
        )

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
            input_dim=tokenizer.vocabulary_size(),
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
            include_preprocessing=include_preprocessing,
        )
        # All references to `self` below this line
        self.tokenizer = tokenizer
        self.packer = packer
        self.token_embedding = token_embedding_layer
        self.vocabulary = vocabulary
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_sequence_length = max_sequence_length
        self.num_segments = num_segments
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.cls_token_index = cls_token_index

    def preprocess_features(self, x):
        # TODO(mattdangerw): figure out how to remove this line
        if isinstance(x, dict):
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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary": self.vocabulary,
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


class BertClassifier(PipelineModel):
    """Bert encoder model with a classification head.

    Args:
        base_model: A `keras_nlp.models.BertCustom` to encode inputs.
        num_classes: Int. Number of classes to predict.
        name: String, optional. Name of the model.
        trainable: Boolean, optional. If the model's variables should be
            trainable.

    Example usage:
    ```python
    # Randomly initialized Bert encoder
    model = keras_nlp.models.BertCustom(
        vocabulary="vocab.txt",
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12
    )

    # Call classifier on the inputs.
    input_data = {
        "input_ids": tf.random.uniform(
            shape=(1, 12), dtype=tf.int64, maxval=30522
        ),
        "segment_ids": tf.constant(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
        "input_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
    }
    classifier = bert.BertClassifier(model, 4, name="classifier")
    logits = classifier(input_data)
    ```
    """

    def __init__(
        self,
        base_model,
        num_classes,
        name=None,
        trainable=True,
        include_preprocessing=True,
    ):
        inputs = base_model.input
        pooled = base_model(inputs, include_preprocessing=False)[
            "pooled_output"
        ]
        outputs = keras.layers.Dense(
            num_classes,
            kernel_initializer=_bert_kernel_initializer(),
            name="logits",
        )(pooled)
        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            name=name,
            trainable=trainable,
            include_preprocessing=include_preprocessing,
        )
        # All references to `self` below this line
        self.base_model = base_model
        self.num_classes = num_classes

    def preprocess_features(self, x):
        return self.base_model.preprocess_features(x)


MODEL_DOCSTRING = """Bert model using "{type}" architecture.

    This network implements a bi-directional Transformer-based encoder as
    described in ["BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding"](https://arxiv.org/abs/1810.04805). It includes the
    embedding lookups and transformer layers, but not the masked language model
    or classification task networks.

    Args:
        weights: String, optional. Either the name of pre-trained model to load
            weights, a path to a weights file, or None.
            For a pre-trained model, weights can be one of {names}.
            If `None`, model is randomly initialized.
        vocabulary: Either the name of a pre-trained model vocabulary, a list
            of vocabulary terms, or filename.
            For a pre-trained model, vocabulary can be one of {names}.
            If `weights` is a pre-trained model name, this argument should
            not be set. The `vocabulary` will be inferred to match the
            pre-trained weights.
        name: String, optional. Name of the model.
        trainable: Boolean, optional. If the model's variables should be
            trainable.

    Example usage:
    ```python
    # Randomly initialized BertBase encoder
    model = keras_nlp.models.BertBase(vocabulary="vocab.txt")

    # Call encoder on the inputs.
    input_data = {{
        "input_ids": tf.random.uniform(
            shape=(1, 512), dtype=tf.int64, maxval=model.vocabulary
        ),
        "segment_ids": tf.constant([0] * 200 + [1] * 312, shape=(1, 512)),
        "input_mask": tf.constant([1] * 512, shape=(1, 512)),
    }}
    output = model(input_data)

    # Load a pretrained model
    model = keras_nlp.models.BertBase(weights="uncased_en")
    # Call encoder on the inputs.
    output = model(input_data)
    ```
"""


def BertBase(weights=None, vocabulary=None, name=None, trainable=True):
    if weights in checkpoints["bert_base"]:
        if vocabulary is not None:
            raise ValueError(
                "When `weights` is set to a pretrained model name."
                "Vocabulary should not be set. Received: "
                f"weights={weights}, vocabulary={vocabulary}."
            )
        # Set the vocabulary name to match the weights.
        vocabulary = weights
        weights = keras.utils.get_file(
            "model.h5",
            BASE_PATH + "bert_base_" + weights + "/model.h5",
            cache_subdir="models/bert_base/" + weights + "/",
            file_hash=checkpoints["bert_base"][weights]["md5"],
        )

    if vocabulary in checkpoints["bert_base"]:
        vocabulary = keras.utils.get_file(
            "vocab.txt",
            BASE_PATH + "bert_base_" + vocabulary + "/vocab.txt",
            cache_subdir="models/bert_base/" + vocabulary + "/",
        )

    model = BertCustom(
        vocabulary=vocabulary,
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
    if weights is not None:
        model.load_weights(weights)

    return model


setattr(
    BertBase,
    "__doc__",
    MODEL_DOCSTRING.format(
        type="Base", names=", ".join(checkpoints["bert_base"])
    ),
)
