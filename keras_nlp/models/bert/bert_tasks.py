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
"""BERT task specific models and heads."""

from tensorflow import keras

from keras_nlp.layers.mlm_head import MLMHead
from keras_nlp.models.bert.bert_models import Bert
from keras_nlp.models.bert.bert_models import bert_kernel_initializer
from keras_nlp.models.bert.bert_preprocessing import BertClassifierPreprocessor
from keras_nlp.models.bert.bert_preprocessing import BertPretrainerPreprocessor
from keras_nlp.models.bert.bert_preprocessing import BertTokenizer
from keras_nlp.utils.pipeline_model import PipelineModel

# TODO(jbischof): Find more scalable way to list checkpoints.
CLASSIFIER_DOCSTRING = """BERT encoder model with a classification head.

    Args:
        backbone: A string or `keras_nlp.models.Bert` instance. If a string,
            should be one of {names}.
        num_classes: int. Number of classes to predict.

    Examples:
    ```python
    # Randomly initialized BERT encoder
    model = keras_nlp.models.BertCustom(
        vocabulary_size=30522,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12
    )

    # Call classifier on the inputs.
    input_data = {{
        "token_ids": tf.random.uniform(
            shape=(1, 12), dtype=tf.int64, maxval=model.vocabulary_size
        ),
        "segment_ids": tf.constant(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
    }}
    classifier = keras_nlp.models.BertClassifier(model, 4, name="classifier")
    logits = classifier(input_data)

    # String backbone specification
    classifier = keras_nlp.models.BertClassifier(
        "bert_base_uncased_en", 4, name="classifier"
    )
    logits = classifier(input_data)

    # Access backbone programatically (e.g., to change `trainable`)
    classifier.backbone.trainable = False
    ```
"""


@keras.utils.register_keras_serializable(package="keras_nlp")
class BertClassifier(PipelineModel):
    def __init__(
        self,
        backbone="bert_base_uncased_en",
        num_classes=2,
        include_preprocessing=True,
        preprocessor=None,
        **kwargs,
    ):
        # Load backbone from string identifier
        # TODO(jbischof): create util function when ready to load backbones in
        # other task classes (e.g., load_backbone_from_string())
        if isinstance(backbone, str):
            if backbone not in Bert.presets:
                raise ValueError(
                    "`backbone` must be one of "
                    f"""{", ".join(Bert.presets)}. Received: {backbone}."""
                )
            preset = backbone
            backbone = Bert.from_preset(preset)
            if include_preprocessing and preprocessor is None:
                tokenizer = BertTokenizer.from_preset(preset)
                preprocessor = BertClassifierPreprocessor(tokenizer)

        inputs = backbone.input
        pooled = backbone(inputs)["pooled_output"]
        outputs = keras.layers.Dense(
            num_classes,
            kernel_initializer=bert_kernel_initializer(),
            activation="softmax",
            name="preds",
        )(pooled)
        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            include_preprocessing=include_preprocessing,
            **kwargs,
        )
        # All references to `self` below this line
        self._backbone = backbone
        self._preprocessor = preprocessor
        self.num_classes = num_classes

        if include_preprocessing and preprocessor is None:
            raise ValueError("Set preprocessor if `include_preprocessing=True`")
        if not include_preprocessing and preprocessor is not None:
            raise ValueError("No preprocessor if `include_preprocessing=False`")

    def preprocess_features(self, x):
        return self.preprocessor(x, return_targets=False)

    def preprocess_samples(self, x, y=None, sample_weight=None):
        return self.preprocessor(x, y=y, sample_weight=sample_weight)

    @property
    def backbone(self):
        """A `keras_nlp.models.Bert` instance providing the preprocessor submodel."""
        return self._backbone

    @property
    def preprocessor(self):
        """A `keras_nlp.models.BertClassifierPreprocessor` for preprocessing."""
        return self._preprocessor

    def get_config(self):
        return {
            "backbone": keras.layers.serialize(self.backbone),
            "num_classes": self.num_classes,
            "name": self.name,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        if "backbone" in config:
            config["backbone"] = keras.layers.deserialize(config["backbone"])
        return cls(**config)

    def compile(
        self,
        optimizer=None,
        loss="sparse_categorical_crossentropy",
        weighted_metrics=["accuracy"],
        jit_compile=True,
        **kwargs,
    ):
        # TODO: We should be able to use `metrics`, not `weighted_metrics`.
        # There is an upstream bug.
        if optimizer is None:
            optimizer = keras.optimizers.experimental.AdamW(5e-5)
        return super().compile(
            optimizer=optimizer,
            loss=loss,
            weighted_metrics=weighted_metrics,
            jit_compile=jit_compile,
            **kwargs,
        )


setattr(
    BertClassifier,
    "__doc__",
    CLASSIFIER_DOCSTRING.format(names=", ".join(Bert.presets)),
)


@keras.utils.register_keras_serializable(package="keras_nlp")
class BertPretrainer(PipelineModel):
    def __init__(
        self,
        backbone="bert_base_uncased_en",
        include_preprocessing=True,
        preprocessor=None,
        **kwargs,
    ):
        # Load backbone from string identifier
        # TODO(jbischof): create util function when ready to load backbones in
        # other task classes (e.g., load_backbone_from_string())
        if isinstance(backbone, str):
            if backbone not in Bert.presets:
                raise ValueError(
                    "`backbone` must be one of "
                    f"""{", ".join(Bert.presets)}. Received: {backbone}."""
                )
            preset = backbone
            backbone = Bert.from_preset(preset)
            if include_preprocessing and preprocessor is None:
                tokenizer = BertTokenizer.from_preset(preset)
                preprocessor = BertPretrainerPreprocessor(tokenizer)

        inputs = {
            **backbone.input,
            "mask_positions": keras.Input(
                shape=(None,), dtype="int32", name="mask_positions"
            ),
        }
        backbone_outputs = backbone(backbone.input)
        mlm_outputs = MLMHead(
            vocabulary_size=backbone.vocabulary_size,
            embedding_weights=backbone.token_embedding.embeddings,
            kernel_initializer=bert_kernel_initializer(),
            activation="softmax",
            name="mlm",
        )(backbone_outputs["sequence_output"], inputs["mask_positions"])
        nsp_outputs = keras.layers.Dense(
            2,
            kernel_initializer=bert_kernel_initializer(),
            activation="softmax",
            name="nsp",
        )(backbone_outputs["pooled_output"])
        outputs = {
            "mlm": mlm_outputs,
            "nsp": nsp_outputs,
        }
        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            include_preprocessing=include_preprocessing,
            **kwargs,
        )
        # All references to `self` below this line
        self._backbone = backbone
        self._preprocessor = preprocessor

        if include_preprocessing and preprocessor is None:
            raise ValueError(
                "Need preprocessor if `include_preprocessing=True`"
            )
        if not include_preprocessing and preprocessor is not None:
            raise ValueError("No preprocessor if `include_preprocessing=False`")

    def preprocess_features(self, x):
        return self.preprocessor(x, return_targets=False)

    def preprocess_samples(self, x, y=None, sample_weight=None):
        return self.preprocessor(x, y=y, sample_weight=sample_weight)

    @property
    def backbone(self):
        """A `keras_nlp.models.Bert` instance providing the preprocessor submodel."""
        return self._backbone

    @property
    def preprocessor(self):
        """A `keras_nlp.models.BertClassifierPreprocessor` for preprocessing."""
        return self._preprocessor

    def get_config(self):
        return {
            "backbone": keras.layers.serialize(self.backbone),
            "num_classes": self.num_classes,
            "name": self.name,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        if "backbone" in config:
            config["backbone"] = keras.layers.deserialize(config["backbone"])
        return cls(**config)

    def compile(
        self,
        optimizer=None,
        loss=None,
        weighted_metrics=None,
        jit_compile=True,
        **kwargs,
    ):
        # TODO: We should be able to use `metrics`, not `weighted_metrics`.
        # There is an upstream bug.
        if optimizer is None:
            optimizer = keras.optimizers.experimental.AdamW(1e-4)
        if loss is None:
            loss = {
                "mlm": keras.losses.SparseCategoricalCrossentropy(),
                "nsp": keras.losses.SparseCategoricalCrossentropy(),
            }
        if weighted_metrics is None:
            weighted_metrics = {
                "mlm": keras.metrics.SparseCategoricalAccuracy(),
                "nsp": keras.metrics.SparseCategoricalAccuracy(),
            }
        return super().compile(
            optimizer=optimizer,
            loss=loss,
            weighted_metrics=weighted_metrics,
            jit_compile=jit_compile,
            **kwargs,
        )
