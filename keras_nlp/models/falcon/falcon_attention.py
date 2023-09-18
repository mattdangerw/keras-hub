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
from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.layers.modeling.rotary_embedding import RotaryEmbedding


@keras.saving.register_keras_serializable(package="keras_nlp")
class FalconAttention(keras.layers.Layer):
    def __init__(self, num_heads):
        self.num_heads = num_heads

    def build(
        self,
        query_shape,
        value_shape,
        key_shape=None,
    ):
        # Einsum variables:
        # b = batch size
        # q = query length
        # k = key/value length
        # m = model dim
        # n = num heads
        # h = head dim
        model_dim = query_shape[-1]
        head_dim = model_dim // self.num_heads
        key_shape = value_shape if key_shape is None else key_shape
        self.query_dense = keras.layers.EinsumDense(
            "bqm,mnh->bqnh",
            output_shape=(None, self.num_heads, head_dim),
            kernel_initializer="variance_scaling",
            name="query_dense",
        )
        self.query_dense.build(query_shape)
        self.value_dense = keras.layers.EinsumDense(
            "bkm,mh->bkh",
            output_shape=(None, head_dim),
            kernel_initializer="variance_scaling",
            name="value_dense",
        )
        self.value_dense.build(value_shape)
        self.key_dense = keras.layers.EinsumDense(
            "bkm,mh->bkh",
            output_shape=(None, head_dim),
            kernel_initializer="variance_scaling",
            name="key_dense",
        )
        self.key_dense.build(key_shape)
        self._dot_product_equation = "bqnh,bkh->bnqk"
        self._combine_equation = "bnqk,bkh->bqnh"
        self.output_dense = keras.layers.EinsumDense(
            "bqnh,nhm->bqm",
            output_shape=(None, model_dim),
            name="output_dense",
        )
        self.output_dense.build((None, None, self.num_heads, head_dim))
        self.rotary_embedding = RotaryEmbedding()
        self.softmax_layer = keras.layers.Softmax()
        self.built = True

    def call(
        self,
        inputs,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
    ):
        query = self._query_dense(inputs)

        if cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            if cache_update_index is None:
                key = key_cache
                value = value_cache
            else:
                key_update = self.key_dense(inputs)
                value_update = self.value_dense(inputs)
                start = [0, cache_update_index, 0, 0]
                key = ops.slice_update(key_cache, start, key_update)
                value = ops.slice_update(value_cache, start, value_update)
                cache = ops.stack((key, value), axis=1)
        else:
            key = self.key_dense(inputs)
            value = self.value_dense(inputs)

        query = self.rotary_embedding(query, start_index=cache_update_index)
        key = self.rotary_embedding(key)
        query = query * ops.rsqrt(ops.cast(ops.shape(query)[-1], query.dtype))
        scores = ops.einsum(self._dot_product_equation, key, query)
        scores = self.softmax_layer(scores, attention_mask)
        outputs = ops.einsum(self._combine_equation, scores, value)
        outputs = self.output_dense(outputs)

        return outputs, cache
