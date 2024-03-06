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

import os

os.environ["KERAS_BACKEND"] = "jax"
# No GPU for conversion, makes memory management easier.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import kagglehub  # noqa: E402
import numpy as np  # noqa: E402
import sentencepiece  # noqa: E402
from absl import app  # noqa: E402
from absl import flags  # noqa: E402
from gemma import params as params_lib  # noqa: E402
from gemma import sampler as sampler_lib  # noqa: E402
from gemma import transformer as transformer_lib  # noqa: E402

import keras_nlp  # noqa: E402

FLAGS = flags.FLAGS

PRESET_MAP = {
    "gemma_2b_en": "google/gemma/flax/2b",
    "gemma_7b_en": "google/gemma/flax/7b",
    "gemma_instruct_2b_en": "google/gemma/flax/2b-it",
    "gemma_instruct_7b_en": "google/gemma/flax/7b-it",
}


flags.DEFINE_string(
    "preset", None, f'Must be one of {",".join(PRESET_MAP.keys())}'
)
flags.mark_flag_as_required("preset")


def download_flax_model(handle):
    return kagglehub.model_download(handle)


def convert_model(flax_config, vocab_size):
    return keras_nlp.models.GemmaBackbone(
        vocabulary_size=vocab_size,
        num_layers=flax_config.num_layers,
        num_query_heads=flax_config.num_heads,
        num_key_value_heads=flax_config.num_kv_heads,
        hidden_dim=flax_config.embed_dim,
        intermediate_dim=flax_config.hidden_dim * 2,
        head_dim=flax_config.head_dim,
    )


def convert_tokenizer(proto_path):
    return keras_nlp.models.GemmaTokenizer(proto=proto_path)


def convert_weights(keras_model, flax_config, flax_params):
    embeddings = flax_params["transformer"]["embedder"]["input_embedding"]
    embeddings = np.asarray(embeddings[:keras_model.vocabulary_size, :])
    keras_model.get_layer("token_embedding").set_weights([embeddings])
    keras_model.get_layer("final_normalization").set_weights(
        [np.asarray(flax_params["transformer"]["final_norm"]["scale"])]
    )
    for i in range(flax_config.num_layers):
        flax_layer_name = f"layer_{i}"
        keras_block = keras_model.get_layer(f"decoder_block_{i}")

        flax_block = flax_params["transformer"][flax_layer_name]
        keras_block.pre_attention_norm.set_weights(
            [flax_block["pre_attention_norm"]["scale"]]
        )
        keras_block.pre_ffw_norm.set_weights(
            [flax_block["pre_ffw_norm"]["scale"]]
        )

        keras_block.gating_ffw.set_weights(
            [flax_block["mlp"]["gating_einsum"][0]]
        )
        keras_block.gating_ffw_2.set_weights(
            [flax_block["mlp"]["gating_einsum"][1]]
        )
        keras_block.ffw_linear.set_weights([flax_block["mlp"]["linear"]])

        attn_block = flax_block["attn"]
        keras_block.attention.query_dense.kernel.assign(
            np.asarray(attn_block["q_einsum"]["w"][:, :, :])
        )
        keras_block.attention.key_dense.kernel.assign(
            np.asarray(attn_block["kv_einsum"]["w"][0, :, :, :])
        )
        keras_block.attention.value_dense.kernel.assign(
            np.asarray(attn_block["kv_einsum"]["w"][1, :, :, :])
        )
        keras_block.attention.output_dense.kernel.assign(
            flax_block["attn"]["attn_vec_einsum"]["w"]
        )


def validate_output(
    flax_params,
    keras_model,
    keras_tokenizer,
):
    input_str = ["the quick brown fox ran, galloped and jumped."]

    # KerasNLP
    token_ids = keras_tokenizer(input_str)
    keras_model_input = {
        "token_ids": np.array(token_ids),
        "padding_mask": np.ones_like(token_ids),
    }
    keras_model_outputs = keras_model(keras_model_input)

    # Comparing the outputs.
    print("🔶 KerasNLP output:", keras_model_outputs[0, 0, :10])
    # print("🔶 HF output:", hf_model_outputs[0, 0, :10])
    # print("🔶 Difference:", np.mean(keras_model_outputs - hf_model_outputs))


def main(_):
    preset = FLAGS.preset

    assert (
        preset in PRESET_MAP.keys()
    ), f'Invalid preset {preset}. Must be one of {",".join(PRESET_MAP.keys())}'

    print(f"✅ Coverting {preset}")

    handle = PRESET_MAP[preset]
    flax_dir = download_flax_model(handle)
    print("✅ Flax model downloaded from kaggle")

    variant = handle.split("/")[-1]
    flax_params = params_lib.load_and_format_params(flax_dir + "/" + variant)
    proto_path = flax_dir  + "/tokenizer.model"
    flax_config = transformer_lib.TransformerConfig.from_params(flax_params)
    print("✅ Flax model loaded")

    keras_tokenizer = convert_tokenizer(proto_path)
    vocab_size = keras_tokenizer.vocabulary_size()
    keras_model = convert_model(flax_config, vocab_size)
    print("✅ Keras model loaded")

    convert_weights(keras_model, flax_config, flax_params)
    print("✅ Weights converted")

    validate_output(
        flax_params,
        keras_model,
        keras_tokenizer,
    )
    print("✅ Numerics validated")

    keras_nlp.src.utils.preset_utils.save_to_preset(keras_model, preset)
    keras_nlp.src.utils.preset_utils.save_to_preset(
        keras_tokenizer, preset, config_filename="tokenizer.json"
    )
    print("✅ Preset saved")


if __name__ == "__main__":
    app.run(main)
