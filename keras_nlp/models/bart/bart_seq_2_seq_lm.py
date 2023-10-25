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

import copy

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.models.bart.bart_backbone import BartBackbone
from keras_nlp.models.bart.bart_presets import backbone_presets
from keras_nlp.models.bart.bart_seq_2_seq_lm_preprocessor import (
    BartSeq2SeqLMPreprocessor,
)
from keras_nlp.models.generative_task import GenerativeTask
from keras_nlp.utils.python_utils import classproperty


@keras_nlp_export("keras_nlp.models.BartSeq2SeqLM")
class BartSeq2SeqLM(GenerativeTask):
    """An end-to-end BART model for seq2seq language modeling.

    A seq2seq language model (LM) is an encoder-decoder model which is used for
    conditional text generation. The encoder is given a "context" text (fed to
    the encoder), and the decoder predicts the next token based on both the
    encoder inputs and the previous tokens. You can finetune `BartSeq2SeqLM` to
    generate text for any seq2seq task (e.g., translation or summarization).

    This model has a `generate()` method, which generates text based on
    encoder inputs and an optional prompt for the decoder. The generation
    strategy used is controlled by an additional `sampler` argument passed to
    `compile()`. You can recompile the model with different `keras_nlp.samplers`
    objects to control the generation. By default, `"top_k"` sampling will be
    used.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to string inputs during
    `fit()`, `predict()`, `evaluate()` and `generate()`. This is done by default
    when creating the model with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/facebookresearch/fairseq/).

    Args:
        backbone: A `keras_nlp.models.BartBackbone` instance.
        preprocessor: A `keras_nlp.models.BartSeq2SeqLMPreprocessor` or `None`.
            If `None`, this model will not apply preprocessing, and inputs
            should be preprocessed before calling the model.

    Examples:

    Use `generate()` to do text generation, given an input context.
    ```python
    bart_lm = keras_nlp.models.BartSeq2SeqLM.from_preset("bart_base_en")
    bart_lm.generate("The quick brown fox", max_length=30)

    # Generate with batched inputs.
    bart_lm.generate(["The quick brown fox", "The whale"], max_length=30)
    ```

    Compile the `generate()` function with a custom sampler.
    ```python
    bart_lm = keras_nlp.models.BartSeq2SeqLM.from_preset("bart_base_en")
    bart_lm.compile(sampler="greedy")
    bart_lm.generate("The quick brown fox", max_length=30)
    ```

    Use `generate()` with encoder inputs and an incomplete decoder input (prompt).
    ```python
    bart_lm = keras_nlp.models.BartSeq2SeqLM.from_preset("bart_base_en")
    bart_lm.generate(
        {
            "encoder_text": "The quick brown fox",
            "decoder_text": "The fast"
        }
    )
    ```

    Use `generate()` without preprocessing.
    ```python
    # Preprocessed inputs, with encoder inputs corresponding to
    # "The quick brown fox", and the decoder inputs to "The fast". Use
    # `"padding_mask"` to indicate values that should not be overridden.
    prompt = {
        "encoder_token_ids": np.array([[0, 133, 2119, 6219, 23602, 2, 1, 1]]),
        "encoder_padding_mask": np.array(
            [[True, True, True, True, True, True, False, False]]
        ),
        "decoder_token_ids": np.array([[2, 0, 133, 1769, 2, 1, 1]]),
        "decoder_padding_mask": np.array([[True, True, True, True, False, False]])
    }

    bart_lm = keras_nlp.models.BartSeq2SeqLM.from_preset(
        "bart_base_en",
        preprocessor=None,
    )
    bart_lm.generate(prompt)
    ```

    Call `fit()` on a single batch.
    ```python
    features = {
        "encoder_text": ["The quick brown fox jumped.", "I forgot my homework."],
        "decoder_text": ["The fast hazel fox leapt.", "I forgot my assignment."]
    }
    bart_lm = keras_nlp.models.BartSeq2SeqLM.from_preset("bart_base_en")
    bart_lm.fit(x=features, batch_size=2)
    ```

    Call `fit()` without preprocessing.
    ```python
    x = {
        "encoder_token_ids": np.array([[0, 133, 2119, 2, 1]] * 2),
        "encoder_padding_mask": np.array([[1, 1, 1, 1, 0]] * 2),
        "decoder_token_ids": np.array([[2, 0, 133, 1769, 2]] * 2),
        "decoder_padding_mask": np.array([[1, 1, 1, 1, 1]] * 2),
    }
    y = np.array([[0, 133, 1769, 2, 1]] * 2)
    sw = np.array([[1, 1, 1, 1, 0]] * 2)

    bart_lm = keras_nlp.models.BartSeq2SeqLM.from_preset(
        "bart_base_en",
        preprocessor=None,
    )
    bart_lm.fit(x=x, y=y, sample_weight=sw, batch_size=2)
    ```

    Custom backbone and vocabulary.
    ```python
    features = {
        "encoder_text": [" afternoon sun"],
        "decoder_text": ["noon sun"],
    }
    vocab = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "Ġafter": 5,
        "noon": 6,
        "Ġsun": 7,
    }
    merges = ["Ġ a", "Ġ s", "Ġ n", "e r", "n o", "o n", "Ġs u", "Ġa f", "no on"]
    merges += ["Ġsu n", "Ġaf t", "Ġaft er"]

    tokenizer = keras_nlp.models.BartTokenizer(
        vocabulary=vocab,
        merges=merges,
    )
    preprocessor = keras_nlp.models.BartSeq2SeqLMPreprocessor(
        tokenizer=tokenizer,
        encoder_sequence_length=128,
        decoder_sequence_length=128,
    )
    backbone = keras_nlp.models.BartBackbone(
        vocabulary_size=50265,
        num_layers=6,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=128,
    )
    bart_lm = keras_nlp.models.BartSeq2SeqLM(
        backbone=backbone,
        preprocessor=preprocessor,
    )
    bart_lm.fit(x=features, batch_size=2)
    ```
    """

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        inputs = backbone.input
        hidden_states = backbone(inputs)["decoder_sequence_output"]
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
        return BartBackbone

    @classproperty
    def preprocessor_cls(cls):
        return BartSeq2SeqLMPreprocessor

    def call_decoder(
        self,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_token_ids,
    ):
        # Embedding layers.
        token_embedding = self.backbone.get_layer("token_embedding")(
            decoder_token_ids
        )
        position_embedding = self.backbone.get_layer(
            "decoder_position_embedding"
        )(token_embedding)

        # Sum, normalize and apply dropout to embeddings.
        x = self.backbone.get_layer("decoder_embeddings_add")(
            (token_embedding, position_embedding)
        )
        x = self.backbone.get_layer("decoder_embeddings_layer_norm")(x)
        x = self.backbone.get_layer("decoder_embeddings_dropout")(x)

        for i in range(self.backbone.num_layers):
            x = self.backbone.get_layer(f"transformer_decoder_layer_{i}")(
                decoder_sequence=x,
                encoder_sequence=encoder_hidden_states,
                encoder_padding_mask=encoder_padding_mask,
            )

        hidden_states = x
        logits = self.backbone.token_embedding(hidden_states, reverse=True)
        return logits, hidden_states

    def call_encoder(self, token_ids, padding_mask):
        # Embedding layers.
        token_embedding = self.backbone.get_layer("token_embedding")(token_ids)
        position_embedding = self.backbone.get_layer(
            "encoder_position_embedding"
        )(token_embedding)

        # Sum, normalize and apply dropout to embeddings.
        x = self.backbone.get_layer("encoder_embeddings_add")(
            (token_embedding, position_embedding)
        )
        x = self.backbone.get_layer("encoder_embeddings_layer_norm")(x)
        x = self.backbone.get_layer("encoder_embeddings_dropout")(x)

        # Transformer encoder layers.
        for i in range(self.backbone.num_layers):
            x = self.backbone.get_layer(f"transformer_encoder_layer_{i}")(
                x, padding_mask=padding_mask
            )

        return x

    def generate_step(
        self,
        inputs,
        end_token_id=None,
    ):
        """A compilable generation function for a batch of inputs.

        This function represents the inner, XLA-compilable, generation function
        for a single batch of inputs. Inputs should have the same structure as
        model inputs, a dictionary with keys `"encoder_token_ids"`,
        `"encoder_padding_mask"`, `"decoder_token_ids"` and
        `"decoder_padding_mask"`.

        Args:
            inputs: A dictionary with four keys - `"encoder_token_ids"`,
                `"encoder_padding_mask"`, `"decoder_token_ids"` and
                `"decoder_padding_mask"`, with batched tensor values.
            end_token_id: The id of the end token to stop on. If all
                sequences have produced a new `end_token_id`, generation
                will stop.
        """
        (
            encoder_token_ids,
            encoder_padding_mask,
            decoder_token_ids,
            decoder_padding_mask,
        ) = (
            inputs["encoder_token_ids"],
            inputs["encoder_padding_mask"],
            inputs["decoder_token_ids"],
            inputs["decoder_padding_mask"],
        )

        batch_size = ops.shape(encoder_token_ids)[0]

        # Create and seed cache with a single forward pass.
        encoder_hidden_states = self.call_encoder(
            token_ids=encoder_token_ids, padding_mask=encoder_padding_mask
        )
        _, hidden_states = self.call_decoder(
            encoder_hidden_states=encoder_hidden_states,
            encoder_padding_mask=encoder_padding_mask,
            decoder_token_ids=decoder_token_ids,
        )
        # Compute the lengths of all user inputted tokens ids.
        row_lengths = ops.sum(ops.cast(decoder_padding_mask, "int32"), axis=-1)
        # Start at the first index that has no user inputted id.
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            num_samples = ops.shape(prompt)[0]

            def repeat_tensor(x):
                """Repeats tensors along batch axis to match dim for beam search."""
                if ops.shape(x)[0] == num_samples:
                    return x
                return ops.repeat(x, repeats=num_samples // batch_size, axis=0)

            logits, hidden_states = self.call_decoder(
                encoder_hidden_states=repeat_tensor(encoder_hidden_states),
                encoder_padding_mask=repeat_tensor(encoder_padding_mask),
                decoder_token_ids=prompt,
            )
            # Slice the last state, this will go away when we add the cache.
            logits = logits[:, index - 1, :]
            hidden_states = hidden_states[:, index - 1, :]
            return logits, hidden_states, cache

        decoder_token_ids = self._sampler(
            next=next,
            prompt=decoder_token_ids,
            index=index,
            mask=decoder_padding_mask,
            end_token_id=end_token_id,
            hidden_states=hidden_states,
        )

        # Compute an output padding mask with the token ids we updated.
        if end_token_id is not None:
            # Build a mask of `end_token_id` locations not in the original
            # prompt (not in locations where `decoder_padding_mask` is True).
            end_locations = ops.logical_and(
                ops.equal(decoder_token_ids, end_token_id),
                ops.logical_not(decoder_padding_mask),
            )
            end_locations = ops.cast(end_locations, "int32")
            # Use cumsum to get ones in all locations after `end_locations`.
            cumsum = ops.cast(ops.cumsum(end_locations, axis=-1), "int32")
            overflow = cumsum - end_locations
            # Our padding mask is the inverse of these overflow locations.
            decoder_padding_mask = ops.logical_not(ops.cast(overflow, "bool"))
        else:
            # Without early stopping, all locations will have been updated.
            decoder_padding_mask = ops.ones_like(
                decoder_token_ids, dtype="bool"
            )

        return {
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }
