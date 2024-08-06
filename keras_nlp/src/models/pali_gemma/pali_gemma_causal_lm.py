# Copyright 2024 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writingf, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from keras import ops

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.causal_lm import CausalLM
from keras_nlp.src.models.pali_gemma.pali_gemma_backbone import (
    PaliGemmaBackbone,
)
from keras_nlp.src.models.pali_gemma.pali_gemma_causal_lm_preprocessor import (
    PaliGemmaCausalLMPreprocessor,
)


@keras_nlp_export("keras_nlp.models.PaliGemmaCausalLM")
class PaliGemmaCausalLM(CausalLM):
    """An end-to-end multi modal PaliGemma model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    image and plain text input, or to autoregressively generate plain text
    similar to the data used for training.

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
        backbone: A `keras_nlp.models.PaliGemmaBackbone` instance.
        preprocessor: A `keras_nlp.models.PaliGemmaCausalLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.

    Examples:

    Use `generate()` to do text generation.
    ```python
    image = np.random.rand(224, 224, 3)
    pali_gemma_lm = keras_nlp.models.PaliGemmaCausalLM.from_preset(
        "pali_gemma_3b_mix_224"
    )
    pali_gemma_lm.generate(
      {
        "images": image,
        "text": ["answer en where is the cow standing?\\n"]
      }
    )

    # Generate with batched prompts.
    pali_gemma_lm.generate(
      {
        "images": [image, image],
        "text": ["answer en where is the cow standing?\\n", "caption en\\n"]
      }
    )
    ```

    Use `generate()` without preprocessing.
    ```python
    image = np.random.rand(224, 224, 3)
    inputs = {
        "images": [image, image],
        # Token ids for "<bos> Keras is".
        "token_ids": np.array([[2, 214064, 603, 0, 0, 0, 0]] * 2),
        # Use `"padding_mask"` to indicate values that should not be overridden.
        "padding_mask": np.array([[1, 1, 1, 0, 0, 0, 0]] * 2),
    }

    pali_gemma_lm = keras_nlp.models.PaliGemmaCausalLM.from_preset(
        "pali_gemma_3b_mix_224",
        preprocessor=None,
    )
    pali_gemma_lm.generate(inputs)
    ```

    Custom backbone and vocabulary.
    ```python
    tokenizer = keras_nlp.models.PaliGemmaTokenizer(
        proto="proto.spm",
    )
    preprocessor = keras_nlp.models.PaliGemmaCausalLMPreprocessor(
        tokenizer=tokenizer,
        sequence_length=128,
    )
    backbone = keras_nlp.models.PaliGemmaBackbone()
    pali_gemma_lm = keras_nlp.models.PaliGemmaCausalLM(
        backbone=backbone,
        preprocessor=preprocessor,
    )
    ```
    """

    backbone_cls = PaliGemmaBackbone
    preprocessor_cls = PaliGemmaCausalLMPreprocessor

    def __init__(
        self,
        preprocessor,
        backbone,
        **kwargs,
    ):
        # === Layers ===
        self.preprocessor = preprocessor
        self.backbone = backbone

        # === Functional Model ===
        inputs = backbone.inputs
        hidden_state = backbone(inputs=inputs)
        outputs = backbone.token_embedding(hidden_state, reverse=True)
        outputs = outputs[:, backbone.image_sequence_length :, :]
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

    def get_generate_inputs(self, inputs):
        batch_size, length = ops.shape(inputs["token_ids"])
        images = inputs["images"]
        if len(ops.shape(images)) == 3:
            images = ops.expand_dims(images, 0)
        length += self.backbone.image_sequence_length
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_key_value_heads
        head_dim = self.backbone.head_dim
        cache_shape = [batch_size, num_layers, 2, length, num_heads, head_dim]
        cache = ops.zeros(cache_shape, dtype=self.compute_dtype)
        image_embeddings = self.backbone.vit_encoder(images)
        _, cache = self._call_transformer_cached(image_embeddings, cache)
        # All information about the images is now captured in the cache.
        return {
            **inputs,
            "cache": cache,
        }

    def generate_call(self, inputs, index, length=1):
        token_ids = inputs["token_ids"]
        batch_size, max_length = ops.shape(token_ids)
        token_ids = ops.slice(token_ids, (0, index), (batch_size, length))
        cache = inputs["cache"]
        x = self.backbone.token_embedding(token_ids)
        x = x * ops.cast(ops.sqrt(self.backbone.hidden_dim), x.dtype)
        index = index + self.backbone.image_sequence_length
        x, cache = self._call_transformer_cached(x, cache, index)
        x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, {
            **inputs,
            "cache": cache, # Update the cache.
        }

    def _call_transformer_cached(self, x, cache, index=0):
        # Each decoder layer has a cache; we update them separately.
        caches = []
        for i, transformer_layer in enumerate(self.backbone.transformer_layers):
            current_cache = cache[:, i, ...]
            x, next_cache = transformer_layer(
                x,
                cache=current_cache,
                cache_update_index=index,
            )
            caches.append(next_cache)
        cache = ops.stack(caches, axis=1)
        return x, cache
