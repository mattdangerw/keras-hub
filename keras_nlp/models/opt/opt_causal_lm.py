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

import copy

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.models.generative_task import GenerativeTask
from keras_nlp.models.opt.opt_backbone import OPTBackbone
from keras_nlp.models.opt.opt_causal_lm_preprocessor import (
    OPTCausalLMPreprocessor,
)
from keras_nlp.models.opt.opt_presets import backbone_presets
from keras_nlp.utils.python_utils import classproperty


@keras_nlp_export("keras_nlp.models.OPTCausalLM")
class OPTCausalLM(GenerativeTask):
    """An end-to-end OPT model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    plain text input, or to autoregressively generate plain text similar to
    the data used for training. This task can be used for pre-training or
    fine-tuning a GPT-2 model, simply by calling `fit()`.

    This model has a `generate()` method, which generates text based on a
    prompt. The generation strategy used is controlled by an additional
    `sampler` argument on `compile()`. You can recompile the model with
    different `keras_nlp.samplers` objects to control the generation. By
    default, `"top_k"` sampling will be used.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to string inputs during
    `fit()`, `predict()`, `evaluate()` and `generate()`. This is done by default
    when creating the model with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/facebookresearch/fairseq/).

    Args:
        backbone: A `keras_nlp.models.OPTBackbone` instance.
        preprocessor: A `keras_nlp.models.OPTCausalLMPreprocessor` or `None`.
            If `None`, this model will not apply preprocessing, and inputs
            should be preprocessed before calling the model.

    Examples:

    Use `generate()` to do text generation.
    ```python
    opt_lm = keras_nlp.models.OPTCausalLM.from_preset("opt_125m_en")
    opt_lm.generate("I want to say", max_length=30)

    # Generate with batched prompts.
    opt_lm.generate(["This is a", "Where are you"], max_length=30)
    ```

    Compile the `generate()` function with a custom sampler.
    ```python
    opt_lm = keras_nlp.models.OPTCausalLM.from_preset("opt_125m_en")
    opt_lm.compile(sampler="greedy")
    opt_lm.generate("I want to say", max_length=30)

    opt_lm.compile(sampler=keras_nlp.samplers.BeamSampler(num_beams=2))
    opt_lm.generate("I want to say", max_length=30)
    ```

    Use `generate()` without preprocessing.
    ```python
    # Prompt the model with `5338, 318` (the token ids for `"Who is"`).
    # Use `"padding_mask"` to indicate values that should not be overridden.
    prompt = {
        "token_ids": np.array([[5338, 318, 0, 0, 0]] * 2),
        "padding_mask": np.array([[1, 1, 0, 0, 0]] * 2),
    }

    opt_lm = keras_nlp.models.OPTCausalLM.from_preset(
        "opt_125m_en",
        preprocessor=None,
    )
    opt_lm.generate(prompt)
    ```

    Call `fit()` on a single batch.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    opt_lm = keras_nlp.models.OPTCausalLM.from_preset("opt_125m_en")
    opt_lm.fit(x=features, batch_size=2)
    ```

    Call `fit()` without preprocessing.
    ```python
    x = {
        "token_ids": np.array([[1, 2, 3, 4, 5]] * 2),
        "padding_mask": np.array([[1, 1, 1, 1, 1]] * 2),
    }
    y = np.array([[2, 3, 4, 5, 0]] * 2)
    sw = np.array([[1, 1, 1, 1, 1]] * 2)

    opt_lm = keras_nlp.models.OPTCausalLM.from_preset(
        "opt_base_en",
        preprocessor=None,
    )
    opt_lm.fit(x=x, y=y, sample_weight=sw, batch_size=2)
    ```

    Custom backbone and vocabulary.
    ```python
    features = ["a quick fox.", "a fox quick."]
    vocab = {"<|endoftext|>": 0, "a": 4, "Ġquick": 5, "Ġfox": 6}
    merges = ["Ġ q", "u i", "c k", "ui ck", "Ġq uick"]
    merges += ["Ġ f", "o x", "Ġf ox"]

    tokenizer = keras_nlp.models.OPTTokenizer(
        vocabulary=vocab,
        merges=merges,
    )
    preprocessor = keras_nlp.models.OPTCausalLMPreprocessor(
        tokenizer=tokenizer,
        sequence_length=128,
    )
    model = keras_nlp.models.OPTBackbone(
        vocabulary_size=50265,
        num_layers=4,
        num_heads=4,
        hidden_dim=256,
        intermediate_dim=512,
        max_sequence_length=128,
    )
    opt_lm = keras_nlp.models.OPTCausalLM(
        backbone=backbone,
        preprocessor=preprocessor,
    )
    opt_lm.fit(x=features, batch_size=2)
    ```
    """

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        inputs = backbone.input
        hidden_states = backbone(inputs)
        outputs = backbone.token_embedding(hidden_states, reverse=True)

        # Instantiate using Functional API Model constructor.
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            include_preprocessing=preprocessor is not None,
            **kwargs,
        )
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.generate_function = None
        self._sampler = None

        # Default compilation
        self.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(2e-5),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
            jit_compile=True,
        )

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classproperty
    def backbone_cls(cls):
        return OPTBackbone

    @classproperty
    def preprocessor_cls(cls):
        return OPTCausalLMPreprocessor

    def add_cache_data(self, data):
        data = dict(data)
        batch_size, max_length = ops.shape(data["token_ids"])
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_heads
        head_dim = self.backbone.hidden_dim // self.backbone.num_heads
        shape = [batch_size, num_layers, 2, max_length, num_heads, head_dim]
        data["cache"] = ops.zeros(shape, dtype=self.compute_dtype)
        return data

    def call_with_cache(self, data, index):
        token_ids = data["token_ids"]
        cache = data["cache"]
        x = self.backbone.get_layer("embeddings")(
            token_ids, start_index=index
        )
        # Each decoder layer has a cache; we update them separately.
        caches = []
        for i in range(self.backbone.num_layers):
            current_cache = cache[:, i, ...]
            x, next_cache = self.backbone.get_layer(f"transformer_layer_{i}")(
                x,
                self_attention_cache=current_cache,
                self_attention_cache_update_index=index,
            )
            caches.append(next_cache)
        cache = ops.stack(caches, axis=1)
        x = self.backbone.get_layer("layer_norm")(x)
        hidden_states = x
        logits = self.backbone.token_embedding(hidden_states, reverse=True)
        return logits, hidden_states, cache
