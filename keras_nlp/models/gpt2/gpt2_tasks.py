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

import tensorflow as tf
from tensorflow import keras

from keras_nlp.models.gpt2.gpt2_models import Gpt2
from keras_nlp.models.gpt2.gpt2_preprocessing import (
    Gpt2TextGeneratorPreprocessor,
)
from keras_nlp.models.gpt2.gpt2_preprocessing import Gpt2Tokenizer
from keras_nlp.utils.pipeline_model import PipelineModel
from keras_nlp.utils.text_generation import random_search


@keras.utils.register_keras_serializable(package="keras_nlp")
class Gpt2TextGenerator(PipelineModel):
    def __init__(
        self,
        backbone="gpt2_base",
        include_preprocessing=True,
        preprocessor=None,
        **kwargs,
    ):
        # Load backbone from string identifier
        if isinstance(backbone, str):
            if backbone not in Gpt2.presets:
                raise ValueError(
                    "`backbone` must be one of "
                    f"""{", ".join(Gpt2.presets)}. Received: {backbone}."""
                )
            preset = backbone
            backbone = Gpt2.from_preset(preset)
            if include_preprocessing and preprocessor is None:
                tokenizer = Gpt2Tokenizer.from_preset(preset)
                preprocessor = Gpt2TextGeneratorPreprocessor(tokenizer)

        inputs = backbone.input
        outputs = backbone(inputs)
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
        """A `keras_nlp.models.Gpt2TextGeneratorPreprocessor` for preprocessing."""
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
            optimizer = keras.optimizers.experimental.AdamW(1e-4)
        return super().compile(
            optimizer=optimizer,
            loss=loss,
            weighted_metrics=weighted_metrics,
            jit_compile=jit_compile,
            **kwargs,
        )

    def predict(
        self,
        x,
        max_length=512,
    ):
        # TODO: This is a real quick and hacky version of generation to show
        # what we could do here.
        @tf.function(jit_compile=True)
        def generate_ids(token_ids):
            def next_token(x, index):
                return self(
                    {
                        "token_ids": x,
                        "padding_mask": x != 0,
                    },
                    include_preprocessing=False,
                )[:, index, :]

            return random_search(
                next_token,
                token_ids,
                max_length=max_length,
            )

        token_ids = self.preprocessor(x, return_targets=False)["token_ids"]
        token_ids = generate_ids(token_ids)
        token_ids = tf.ragged.boolean_mask(token_ids, token_ids != 0)
        return self.preprocessor.tokenizer.detokenize(token_ids)
