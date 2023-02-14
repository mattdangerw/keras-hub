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
"""GPT2 Causal LM (Language Model)."""

import copy

import tensorflow as tf
from tensorflow import keras

import keras_nlp
from keras_nlp.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_nlp.models.gpt2.gpt2_causal_lm_preprocessor import (
    GPT2CausalLMPreprocessor,
)
from keras_nlp.models.gpt2.gpt2_presets import backbone_presets
from keras_nlp.models.task import Task
from keras_nlp.utils.python_utils import classproperty


@keras.utils.register_keras_serializable(package="keras_nlp")
class GPT2CausalLM(Task):
    """An end-to-end GPT2 model for causal langauge modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens the next token based on previous tokens, which is the way GPT2 gets
    pretrained. You can finetune `GPT2CausalLM` to generate text similar to
    the custom dataset. `GPT2CausalLM` also has a method `generate()`, which
    generates text based on given prompt.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to raw inputs during
    `fit()`, `predict()`, and `evaluate()`. This is done by default when
    creating the model with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/openai/gpt-2).

    Args:
        backbone: A `keras_nlp.models.GPT2Backbone` instance.
        preprocessor: A `keras_nlp.models.GPT2CausalLMPreprocessor` or `None`.
            If `None`, this model will not apply preprocessing, and inputs
            should be preprocessed before calling the model.

    Examples:

    Use `generate()` method to do text generation.
    ```python
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en")
    gpt2_lm.generate("I want to say", max_length=30)

    # Generate with batched prompts.
    gpt2_lm.generate(["This is a", "Where are you"], max_length=30)
    ```

    Use a custom sampler for text generation.
    ```python
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en")

    # Use string identifier to set sampler.
    gpt2_lm.generate("I want to say", max_length=30, sampler="top_p")

    # Construct a sampler instance.
    sampler = keras_nlp.samplers.BeamSampler(num_beams=2)
    gpt2_lm.generate("I want to say", max_length=30, sampler=sampler)
    ```

    Map raw string to languages model logit predictions.
    ```python
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en")
    gpt2_lm.predict(["You know this is just a test string"])
    ```

    Load a pretrained GPT2 and fit on a string dataset.
    ```python
    features = [
        "I don't listen to music while coding.",
        "But I watch youtube while coding!",
    ]
    ds = tf.data.Dataset.from_tensor_slices(features)

    # Create a `GPT2CausalLM` and fit your data.
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
        "gpt2_base_en",
    )
    gpt2_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    gpt2_lm.fit(ds, batch_size=2)
    ```

    Load a pretrained `GPT2CausalLM` with custom preprocessor, and predict on
    string inputs.
    ```python
    # Use a shorter sequence length.
    preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
        "gpt2_base_en",
        sequence_length=128,
    )

    # Create a `GPT2CausalLM`, using pretrained GPT2 and custom preprocessor.
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
        "gpt2_base_en",
        preprocessor=preprocessor,
    )
    gpt2_lm.predict(["You know this is still a test string"])
    ```

    Fit your preprocessed data with randomly initialized GPT2. This is useful
    when you want to do data preprocessing inside `tf.data` pipeline.
    ```python
    # Define preprocessed input.
    features = {
        "token_ids": tf.constant(
            [[1, 2, 3, 4, 0, 0]] * 2, shape=(2, 6)
        ),
        "padding_mask": tf.constant(
            [[1, 1, 1, 1, 0, 0]] * 2, shape=(2, 6)
        ),
    }
    labels = tf.constant(
        [[2, 3, 4, 0, 0, 0]] * 2, shape=(2, 6)
    )
    sample_weight = tf.constant(
        [[1, 1, 1, 0, 0, 0]] * 2, shape=(2, 6)
    )

    # Randomly initialize a GPT2 backbone.
    backbone = keras_nlp.models.GPT2Backbone(
        vocabulary_size=50257,
        num_layers=2,
        num_heads=2,
        hidden_dim=128,
        intermediate_dim=256,
        max_sequence_length=128,
    )
    # Create a `GPT2CausalLM` without preprocessor and fit the data.
    gpt2_lm = keras_nlp.models.GPT2CausalLM(backbone, preprocessor=None)
    gpt2_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    gpt2_lm.fit(
        x=features,
        y=labels,
        sample_weight=sample_weight,
        batch_size=2,
    )
    ```

    """

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        inputs = backbone.input
        x = backbone(inputs)
        # Use token embedding weights to project from the token representation
        # to vocabulary logits.
        outputs = tf.matmul(
            x,
            backbone.token_embedding.embeddings,
            transpose_b=True,
        )

        # Instantiate using Functional API Model constructor.
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            include_preprocessing=preprocessor is not None,
            **kwargs,
        )

        self.backbone = backbone
        self.preprocessor = preprocessor

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classproperty
    def backbone_cls(cls):
        return GPT2Backbone

    @classproperty
    def preprocessor_cls(cls):
        return GPT2CausalLMPreprocessor

    def build_cache(self, batch_size, length):
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_heads
        head_dim = self.backbone.hidden_dim // self.backbone.num_heads
        shape = (batch_size, num_layers, 2, length, num_heads, head_dim)
        return tf.zeros(shape)

    def build_model_with_cache(self):
        token_ids = self.backbone.input["token_ids"]
        padding_mask = self.backbone.input["padding_mask"]
        # Cache inputs
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_heads
        head_dim = self.backbone.hidden_dim // self.backbone.num_heads
        input_cache = keras.Input(
            shape=(num_layers, 2, None, num_heads, head_dim),
            name="cache",
        )
        cache_index = keras.Input(
            batch_input_shape=(), dtype="int32", name="cache_index"
        )
        # Embed tokens, positions.
        token_embedding = self.backbone.get_layer("token_embedding")(token_ids)
        position_embedding = self.backbone.get_layer("position_embedding")(
            token_embedding,
            start_index=cache_index,
        )
        # Sum and apply dropout to embeddings.
        x = self.backbone.get_layer("embeddings_sum")(
            (token_embedding, position_embedding),
        )
        x = self.backbone.get_layer("embeddings_dropout")(x)
        # Apply successive transformer decoder blocks with a cache.
        output_cache = []
        for i in range(self.backbone.num_layers):
            x, cache = self.backbone.get_layer(f"transformer_layer_{i}")(
                x,
                decoder_padding_mask=padding_mask,
                cache=input_cache[:, i, ...],
                cache_index=cache_index,
            )
            output_cache.append(cache)
        output_cache = tf.stack(output_cache, axis=1)
        # Final layer norm.
        x = self.backbone.get_layer("layer_norm")(x)
        # Language model logits.
        outputs = tf.matmul(
            x,
            self.backbone.token_embedding.embeddings,
            transpose_b=True,
        )
        return keras.Model(
            inputs={
                "token_ids": token_ids,
                "padding_mask": padding_mask,
                "cache": input_cache,
                "cache_index": cache_index,
            },
            outputs=(outputs, output_cache),
        )

    def generate(
        self,
        prompt,
        max_length,
        sampler="top_k",
    ):
        """Generate text.

        This method generates text based on given `prompt`. Generation will
        continue until `max_length` is met, and all tokens generated after
        `end_token` will be truncated. The sampling approach used can be
        controlled via the sampler argument.

        Args:
            prompt: a string, string Tensor or string RaggedTensor. The prompt
                text for generation.
            max_length: int. The max length of generated sequence.
            sampler: a string or `keras_nlp.samplers.Sampler` instance. The
                sampler to be used for text generation.
        """
        sampler = keras_nlp.samplers.get(sampler)
        if hasattr(self, "jit_compile"):
            # `jit_compile` is a public property as of tf 2.12. hasattr is for
            # backward compat.
            sampler.jit_compile = self.jit_compile
        sampler.run_eagerly = self.run_eagerly

        model_with_cache = self.build_model_with_cache()

        def next_token_fn(prompt, mask, cache, cache_index):
            return model_with_cache(
                {
                    "token_ids": prompt,
                    "padding_mask": mask,
                    "cache": cache,
                    "cache_index": cache_index,
                }
            )

        prompt = self.preprocessor.tokenizer(prompt)
        batch_size = 1 if prompt.shape.rank == 1 else prompt.shape.as_list()[0]
        generated = sampler(
            prompt,
            next_token_fn,
            max_length=max_length,
            end_token_id=self.preprocessor.tokenizer.end_token_id,
            cache=self.build_cache(batch_size, max_length),
        )
        return self.preprocessor.tokenizer.detokenize(generated)
