# Copyright 2024 The KerasNLP Authors
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


import keras
from keras import ops

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.causal_lm import CausalLM
from keras_nlp.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_nlp.src.models.gemma.gemma_causal_lm_preprocessor import (
    GemmaCausalLMPreprocessor,
)


@keras_nlp_export("keras_nlp.models.GemmaCausalLM")
class GemmaCausalLM(CausalLM):
    """An end-to-end Gemma model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    plain text input, or to autoregressively generate plain text similar to
    the data used for training. This task can be used for pre-training or
    fine-tuning a Gemma model, simply by calling `fit()`.

    This model has a `generate()` method, which generates text based on a
    prompt. The generation strategy used is controlled by an additional
    `sampler` argument on `compile()`. You can recompile the model with
    different `keras_nlp.samplers` objects to control the generation. By
    default, `"greedy"` sampling will be used.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to string inputs during
    `fit()`, `predict()`, `evaluate()` and `generate()`. This is done by default
    when creating the model with `from_preset()`.

    Args:
        backbone: A `keras_nlp.models.GemmaBackbone` instance.
        preprocessor: A `keras_nlp.models.GemmaCausalLMPreprocessor` or `None`.
            If `None`, this model will not apply preprocessing, and inputs
            should be preprocessed before calling the model.

    Examples:

    Use `generate()` to do text generation.
    ```python
    gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
    gemma_lm.generate("I want to say", max_length=30)

    # Generate with batched prompts.
    gemma_lm.generate(["This is a", "Where are you"], max_length=30)
    ```

    Compile the `generate()` function with a custom sampler.
    ```python
    gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
    gemma_lm.compile(sampler="top_k")
    gemma_lm.generate("I want to say", max_length=30)

    gemma_lm.compile(sampler=keras_nlp.samplers.BeamSampler(num_beams=2))
    gemma_lm.generate("I want to say", max_length=30)
    ```

    Use `generate()` without preprocessing.
    ```python
    prompt = {
        # Token ids for "<bos> Keras is".
        "token_ids": np.array([[2, 214064, 603, 0, 0, 0, 0]] * 2),
        # Use `"padding_mask"` to indicate values that should not be overridden.
        "padding_mask": np.array([[1, 1, 1, 0, 0, 0, 0]] * 2),
    }

    gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(
        "gemma_2b_en",
        preprocessor=None,
    )
    gemma_lm.generate(prompt)
    ```

    Call `fit()` on a single batch.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
    gemma_lm.fit(x=features, batch_size=2)
    ```

    Call `fit()` with LoRA fine-tuning enabled.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
    gemma.backbone.enable_lora(rank=4)
    gemma_lm.fit(x=features, batch_size=2)
    ```

    Call `fit()` without preprocessing.
    ```python
    x = {
        # Token ids for "<bos> Keras is deep learning library<eos>"
        "token_ids": np.array([[2, 214064, 603, 5271, 6044, 9581, 1, 0]] * 2),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 0]] * 2),
    }
    y = np.array([[214064, 603, 5271, 6044, 9581, 3, 0, 0]] * 2)
    sw = np.array([[1, 1, 1, 1, 1, 1, 0, 0]] * 2)

    gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(
        "gemma_2b_en",
        preprocessor=None,
    )
    gemma_lm.fit(x=x, y=y, sample_weight=sw, batch_size=2)
    ```

    Custom backbone and vocabulary.
    ```python
    tokenizer = keras_nlp.models.GemmaTokenizer(
        proto="proto.spm",
    )
    preprocessor = keras_nlp.models.GemmaCausalLMPreprocessor(
        tokenizer=tokenizer,
        sequence_length=128,
    )
    backbone = keras_nlp.models.GemmaBackbone(
        vocabulary_size=30552,
        num_layers=4,
        num_heads=4,
        hidden_dim=256,
        intermediate_dim=512,
        max_sequence_length=128,
    )
    gemma_lm = keras_nlp.models.GemmaCausalLM(
        backbone=backbone,
        preprocessor=preprocessor,
    )
    gemma_lm.fit(x=features, batch_size=2)
    ```
    """

    backbone_cls = GemmaBackbone
    preprocessor_cls = GemmaCausalLMPreprocessor

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        inputs = backbone.input
        hidden_states = backbone(inputs)
        outputs = backbone.token_embedding(hidden_states, reverse=True)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        weighted_metrics="auto",
        sampler="greedy",
        **kwargs,
    ):
        super().compile(
            optimizer=optimizer,
            loss=loss,
            weighted_metrics=weighted_metrics,
            sampler=sampler,
            **kwargs,
        )

    def build_cache(self, batch_size, max_length):
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_key_value_heads
        head_dim = self.backbone.head_dim
        caches = []
        for _ in range(num_layers):
            shape = [batch_size, max_length, num_heads, head_dim]
            key_cache = ops.zeros(shape, dtype=self.compute_dtype)
            value_cache = ops.zeros(shape, dtype=self.compute_dtype)
            caches.append((key_cache, value_cache))
        return tuple(caches)

    def call_with_cache(self, token_ids, caches, index):
        x = self.backbone.token_embedding(token_ids)
        x = x * ops.cast(ops.sqrt(self.backbone.hidden_dim), x.dtype)
        # Each decoder layer has a cache; we update them separately.
        updated_caches = []
        for layer, cache in zip(self.backbone.transformer_layers, caches):
            x, next_cache = layer(
                x,
                cache=cache,
                cache_update_index=index,
            )
            updated_caches.append(next_cache)
        hidden_states = x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, hidden_states, updated_caches

    def score(
        self,
        token_ids,
        padding_mask=None,
        scoring_mode="logits",
        layer_intercept_fn=None,
        target_ids=None,
    ):
        """Score a generation represented by the provided token ids.

        Args:
            token_ids: A <int>[batch_size, num_tokens] tensor containing tokens
                to score. Typically, this tensor captures the output from a call
                to `GemmaCausalLM.generate()`, i.e., tokens for both the input
                text and the model-generated text.
            padding_mask: A <bool>[batch_size, num_tokens] tensor indicating the
                tokens that should be preserved during generation. This is an
                artifact required by the GemmaBackbone and isn't influential on
                the computation of this function. If omitted, this function uses
                `keras.ops.ones()` to create a tensor of the appropriate shape.
            scoring_mode: The type of scores to return, either "logits" or
                "loss", both will be per input token.
            layer_intercept_fn: An optional function for augmenting activations
                with additional computation, for example, as part of
                interpretability research. This function will be passed the
                activations as its first parameter and a numeric index
                associated with that backbone layer. _This index _is not_ an
                index into `self.backbone.layers`_. The index -1 accompanies the
                embeddings returned by calling `self.backbone.token_embedding()`
                on `token_ids` in the forward direction. All subsequent indexes
                will be 0-based indices for the activations returned by each of
                the Transformers layers in the backbone. This function must
                return a <float>[batch_size, num_tokens, hidden_dims] tensor
                that can be passed as an input to the next layer in the model.
            target_ids: An <bool>[batch_size, num_tokens] tensor containing the
                predicted tokens against which the loss should be computed. If a
                span of tokens is provided (sequential truthy values along
                axis=1 in the tensor), the loss will be computed as the
                aggregate across those tokens.

        Raises:
            ValueError: If an unsupported scoring_mode is provided, or if the
                target_ids are not provided when using ScoringMode.LOSS.

        Returns:
            The per-token scores as a tensor of size
            <float>[batch_size, num_tokens, vocab_size] in "logits" mode, or
            <float>[batch_size, num_tokens] in "loss" mode.

        Example:

        Compute gradients between embeddings and loss scores with TensorFlow:
        ```python
        gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(
            "gemma_2b_en"
        )
        generations = gemma_lm.generate(
            ["This is a", "Where are you"],
            max_length=30
        )
        preprocessed = gemma_lm.preprocessor.generate_preprocess(generations)
        generation_ids = preprocessed["token_ids"]
        padding_mask = preprocessed["padding_mask"]
        target_ids = keras.ops.roll(generation_ids, shift=-1, axis=1)

        embeddings = None
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            def layer_intercept_fn(x, i):
                if i == -1:
                    nonlocal embeddings, tape
                    embeddings = x
                    tape.watch(embeddings)
                return x

            losses = gemma_lm.score(
                token_ids=generation_ids,
                padding_mask=padding_mask,
                scoring_mode="loss",
                layer_intercept_fn=layer_intercept_fn,
                target_ids=target_ids,
            )

        grads = tape.gradient(losses, embeddings)
        ```
        """
        if scoring_mode not in ("logits", "loss"):
            raise ValueError(
                "Unsupported scoring_mode. Must be one of 'logits' or 'loss'."
            )

        if scoring_mode == "loss" and target_ids is None:
            raise ValueError(
                "Cannot compute loss without targets. Please provide target "
                "token ids via the target_ids parameter."
            )

        batch_shape = ops.shape(token_ids)[:2]
        assert len(batch_shape) == 2

        if padding_mask is None:
            padding_mask = ops.ones(shape=batch_shape)

        if layer_intercept_fn is None:

            def default_layer_intercept_fn(x, unused_i):
                return x

            layer_intercept_fn = default_layer_intercept_fn

        token_embeddings = self.backbone.token_embedding(token_ids)
        x = layer_intercept_fn(token_embeddings, -1)

        x = token_embeddings * ops.cast(
            ops.sqrt(self.backbone.hidden_dim), dtype=self.compute_dtype
        )
        for i, transformer_layer in enumerate(self.backbone.transformer_layers):
            x = transformer_layer(x, padding_mask=padding_mask)
            x = layer_intercept_fn(x, i)
        x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)

        if scoring_mode == "logits":
            return logits

        per_token_loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        per_token_loss = per_token_loss_fn(target_ids, logits)
        return per_token_loss
