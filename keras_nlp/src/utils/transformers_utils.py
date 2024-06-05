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
"""Convert huggingface models to KerasNLP."""

from keras_nlp.src.utils.preset_utils import get_file


def load_transformers_backbone(cls, preset, load_weights):
    # Check by classname to avoid circular imports. Better way to do this?
    if cls.__name__ == "GemmaBackbone":
        return cls(
            vocabulary_size=256000,
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=1,
            hidden_dim=128,
            intermediate_dim=256,
            head_dim=32,
        )
    raise ValueError(f"No conversion huggingface/transformers to {cls}")


def load_transformers_tokenizer(cls, preset):
    if cls.__name__ == "GemmaTokenizer":
        return cls(get_file(preset, "tokenizer.model"))
    raise ValueError(f"No conversion huggingface/transformers to {cls}")
