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

import copy

from tensorflow import keras

from keras_nlp.models.bert.bert_models import Bert
from keras_nlp.models.bert.bert_models import bert_kernel_initializer
from keras_nlp.models.bert.bert_preprocessing import BertPreprocessor
from keras_nlp.models.bert.bert_presets import backbone_presets
from keras_nlp.models.utils import classproperty
from keras_nlp.utils.pipeline_model import PipelineModel

# TODO(jbischof): Find more scalable way to list checkpoints.
CLASSIFIER_DOCSTRING = """BERT encoder model with a classification head.

    Args:
        backbone: A string or `keras_nlp.models.Bert` instance.
        preprocessor: A `keras_nlp.models.BertPreprocessor`. If `None`, inputs
            will not be preprocessed automatically.
        num_classes: int. Number of classes to predict.

    Examples:
    ```python
    # Randomly initialized BERT backbone.
    model = keras_nlp.models.Bert(
        vocabulary_size=30522,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12
    )
    # Preset preprocessing.
    preprocessor = keras_nlp.models.BertPreprocessor("bert_base_uncased_en")

    # Call classifier on the inputs.
    input_data = tf.constant(["Classify this sentence!"])
    classifier = keras_nlp.models.BertClassifier(
        backbone=model,
        preprocessor=preprocessor,
        num_classes=2)
    logits = classifier.predict(input_data)

    # Use a preset backbone and preprocessing
    classifier = keras_nlp.models.BertClassifier(
        "bert_base_uncased_en",
        num_classes=2,
    )
    logits = classifier.predict(input_data)

    # Access backbone programatically (e.g., to change `trainable`).
    classifier.backbone.trainable = False

    # Access preprocessor programatically (e.g., to run preprocessing).
    classifier.preprocessor("Preprocess this sentence!")
    ```
"""


@keras.utils.register_keras_serializable(package="keras_nlp")
class BertClassifier(PipelineModel):
    def __init__(
        self,
        backbone,
        preprocessor=None,
        num_classes=2,
        **kwargs,
    ):
        inputs = backbone.input
        pooled = backbone(inputs)["pooled_output"]
        outputs = keras.layers.Dense(
            num_classes,
            kernel_initializer=bert_kernel_initializer(),
            name="preds",
        )(pooled)
        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            include_preprocessing=preprocessor is not None,
            **kwargs,
        )
        # All references to `self` below this line
        self._backbone = backbone
        self._preprocessor = preprocessor
        self.num_classes = num_classes

    def preprocess_samples(self, x, y=None, sample_weight=None):
        return self.preprocessor(x, y=y, sample_weight=sample_weight)

    @property
    def backbone(self):
        """A `keras_nlp.models.Bert` instance."""
        return self._backbone

    @property
    def preprocessor(self):
        """A `keras_nlp.models.BertPreprocessor` for preprocessing."""
        return self._preprocessor

    def get_config(self):
        return {
            "backbone": keras.layers.serialize(self.backbone),
            "preprocessor": keras.layers.serialize(self.preprocessor),
            "num_classes": self.num_classes,
            "name": self.name,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        if "backbone" in config:
            config["backbone"] = keras.layers.deserialize(config["backbone"])
        if "preprocessor" in config:
            config["preprocessor"] = keras.layers.deserialize(
                config["preprocessor"]
            )
        return cls(**config)

    def compile(
        self,
        optimizer=None,
        loss=None,
        metrics=None,
        jit_compile=True,
        **kwargs,
    ):
        if optimizer is None:
            optimizer = keras.optimizers.experimental.AdamW(5e-5)
        if loss is None:
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        if metrics is None:
            metrics = keras.metrics.SparseCategoricalAccuracy()
        return super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            jit_compile=jit_compile,
            **kwargs,
        )

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        if "backbone" not in kwargs:
            kwargs["backbone"] = Bert.from_preset(preset)
        if "preprocessor" not in kwargs:
            kwargs["preprocessor"] = BertPreprocessor.from_preset(preset)
        return cls(**kwargs)


setattr(
    BertClassifier,
    "__doc__",
    CLASSIFIER_DOCSTRING,
)
