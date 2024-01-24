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

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.backend import random
from keras_nlp.utils.python_utils import format_docstring

call_args_docstring = """next: A function which takes in the
            `prompt, cache, index` of the current generation loop, and outputs
            a tuple `(logits, hidden_states, cache)` with `logits` being the
            logits of next token, `hidden_states` being the representation of
            the next token, and `cache` for next iteration.
        prompt: A 2D integer tensor with shape `(batch_size, max_length)`. This
            tensor will be iteratively updated column by column with new sampled
            values, starting at `index`.
        cache: Optional. A tensor or nested structure of tensors that will be
            updated by each call to `next`. This can be used to cache
            computations from early iterations of the generative loop.
        index: Optional. The first index of `prompt` to start sampling at.
            Usually this is set as the length of the shortest non-padded
            sequence in `prompt`.
        mask: Optional. A 2D integer tensor with the same shape as `prompt`.
            Locations which are `True` in the mask are never updated during
            sampling. Usually used to mark all locations in the dense prompt
            tensor which were present in a user input.
        end_token_id: Optional. The token marking the end of the sequence. If
            specified, sampling will stop as soon as all sequences in the prompt
            produce a `end_token_id` in a location where `mask` is `False`.
"""


@format_docstring(call_args=call_args_docstring)
@keras_nlp_export("keras_nlp.samplers.Sampler")
class Sampler:
    """Base sampler class.

    Args:
        temperature: float. optional. Used to control the
            randomness of the sampling. The higher the temperature, the
            more diverse the samples. Defaults to `1.0`.

    Call arguments:
        {{call_args}}

    This base class can be extended to implement different auto-regressive
    sampling methods. Subclasses can either:

    - Override the `get_next_token()` method, which computes the next token
      based on a probability distribution over all possible vocab entries.
    - Override `__call__`, if the sampling method needs additional information
      beyond the next tokens probability distribution to sample a sequence.

    Please check available subclass samplers for examples.

    Examples:

    ```python
    # Use a simple alphabet of lowercase characters with ids in range [0, 25].
    int_lookup = {i: chr(i + ord('a')) for i in range(26)}
    char_lookup = {v: k for k, v in int_lookup.items()}
    batch_size, length, vocab_size = 1, 12, len(int_lookup)

    def next(prompt, cache, index):
        # return a uniform distribution over our alphabet.
        logits = ops.ones((batch_size, vocab_size))
        return logits, None, cache

    output = keras_nlp.samplers.GreedySampler()(
        next=next,
        prompt=ops.fill((batch_size, length,), char_lookup['z']),
        index=5,
    )
    print(["".join([int_lookup[i] for i in s]) for s in output.numpy()])
    # >>> ['zzzzzaaaaaaa']
    ```
    """

    def __init__(
        self,
        temperature=1.0,
    ):
        self.temperature = temperature
        self._seed_generators = []

    def __setattr__(self, name, value):
        # We could update to the `Tracker` class from keras-core if our needs
        # become more advanced (e.g. list assignment, nested trackables). For
        # now, we only track `SeedGenerator` instances directly on the sampler.
        if isinstance(value, random.SeedGenerator):
            self._seed_generators.append(value)
        return super().__setattr__(name, value)

    @property
    def variables(self):
        variables = []
        for sg in self._seed_generators:
            variables.append(sg.state)
        return variables

    def start(
        self,
        data,
        logits,
        hidden_states,
        cache,
    ):
        padding_mask = data["padding_mask"]
        # Compute the lengths of all user inputted tokens ids.
        row_lengths = ops.sum(padding_mask, axis=-1)
        # Start at the last index that has all user inputted ids.
        index = ops.min(row_lengths) - 1
        logits = logits[:, index][:, None]
        hidden_states = hidden_states[:, index][:, None]
        return self.next(
            data=data,
            index=index,
            logits=logits,
            hidden_states=hidden_states,
            cache=cache,
        )

    def alive(
        self,
        data,
        index,
        end_token_id=None,
    ):
        token_ids = data["token_ids"]
        padding_mask = data["padding_mask"]
        _, max_length = ops.shape(token_ids)
        length_remaining = ops.less(index, max_length - 1)
        if end_token_id is None:
            return length_remaining
        end_tokens = ops.equal(token_ids, end_token_id)
        end_tokens = ops.logical_and(end_tokens, ops.equal(padding_mask, 2))
        sequence_done = ops.any(end_tokens, axis=-1)
        any_alive = ops.logical_not(ops.all(sequence_done))
        return ops.logical_and(length_remaining, any_alive)

    def next(
        self,
        data,
        index,
        logits,
        hidden_states,
        cache,
    ):
        token_ids = data["token_ids"]
        padding_mask = data["padding_mask"]
        probabilities = self.compute_probabilities(logits)
        # Compute the next token.
        new_tokens = self.get_next_token(probabilities)
        # Update tokens at the following index.
        index = index + 1
        padding_column = padding_mask[:, index][:, None]
        token_column = token_ids[:, index][:, None]
        # Don't overwrite anywhere mask is True.
        new_tokens = ops.cast(new_tokens, token_ids.dtype)
        new_tokens = ops.where(padding_column, token_column, new_tokens)
        new_padding = ops.ones_like(new_tokens) * 2
        new_padding = ops.where(padding_column, padding_column, new_padding)
        # Update the prompt with the next token.
        token_ids = ops.slice_update(token_ids, [0, index], new_tokens)
        padding_mask = ops.slice_update(padding_mask, [0, index], new_padding)
        data["token_ids"] = token_ids
        data["padding_mask"] = padding_mask
        data["cache"] = cache
        return data, index

    def end(
        self,
        data,
    ):
        return data

    def compute_probabilities(self, logits):
        """Compute token probabilities from logits.

        This will always be done in full precision, regardless of dtype, and
        scale by `temperature`.
        """
        logits_dtype = logits.dtype
        logits = ops.cast(logits, "float32")
        probs = keras.activations.softmax(logits / self.temperature)
        return ops.cast(probs, logits_dtype)

    def get_next_token(self, probabilities):
        """Get the next token.
        Args:
            probabilities: a Tensor, the probability distribution for next
                token over all vocab tokens.
        Get the next token based on given probability distribution over tokens.
        Subclasses must implement this method.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {"temperature": self.temperature}
