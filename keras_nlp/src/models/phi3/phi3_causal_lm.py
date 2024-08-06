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
from keras import ops

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.causal_lm import CausalLM
from keras_nlp.src.models.phi3.phi3_backbone import Phi3Backbone
from keras_nlp.src.models.phi3.phi3_causal_lm_preprocessor import (
    Phi3CausalLMPreprocessor,
)
from keras_nlp.src.utils.python_utils import classproperty
from keras_nlp.src.utils.tensor_utils import any_equal


@keras_nlp_export("keras_nlp.models.Phi3CausalLM")
class Phi3CausalLM(CausalLM):
    """An end-to-end Phi3 model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    plain text input, or to autoregressively generate plain text similar to
    the data used for training. This task can be used for pre-training or
    fine-tuning a Phi-3 model, simply by calling `fit()`.

    This model has a `generate()` method, which generates text based on a
    prompt. The generation strategy used is controlled by an additional
    `sampler` argument on `compile()`. You can recompile the model with
    different `keras_nlp.samplers` objects to control the generation. By
    default, `"top_k"` sampling will be used.

    Args:
        backbone: A `keras_nlp.models.Phi3Backbone` instance.
        preprocessor: A `keras_nlp.models.Phi3CausalLMPreprocessor` or `None`.
            If `None`, this model will not apply preprocessing, and inputs
            should be preprocessed before calling the model.
    """

    def __init__(self, backbone, preprocessor=None, **kwargs):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        inputs = backbone.inputs
        hidden_states = backbone(inputs)
        outputs = backbone.token_embedding(hidden_states, reverse=True)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    @classproperty
    def backbone_cls(cls):
        return Phi3Backbone

    @classproperty
    def preprocessor_cls(cls):
        return Phi3CausalLMPreprocessor

    def build_cache(self, batch_size, max_length):
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_heads
        num_heads = self.backbone.num_key_value_heads
        head_dim = self.backbone.hidden_dim // self.backbone.num_query_heads
        shape = [batch_size, num_layers, 2, max_length, num_heads, head_dim]
        return ops.zeros(shape, dtype=self.compute_dtype)

    def call_with_cache(self, token_ids, cache, index):
        x = self.backbone.token_embedding(token_ids)
        # Each decoder layer has a cache; we update them separately.
        updated_cache = []
        for i in range(self.backbone.num_layers):
            current_cache = cache[:, i, ...]
            x, next_cache = self.backbone.transformer_layers[i](
                x,
                attention_cache=current_cache,
                attention_cache_update_index=index,
            )
            updated_cache.append(next_cache)
        cache = ops.stack(updated_cache, axis=1)
        hidden_states = x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, hidden_states, cache
