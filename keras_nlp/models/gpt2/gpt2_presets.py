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

backbone_presets = {
    "gpt2_base": {
        "config": {
            "vocabulary_size": 50257,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "max_sequence_length": 1024,
        },
        "preprocessor_config": {},
        "description": "Gpt2",
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_base/model.h5",
        "weights_hash": "f4ea6e1b214516dd7de452461ee6e16e",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_base/vocab.json",
        "vocabulary_hash": "dffec25a898b1f5e569bec4dffd7e5c0",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_base/merges.txt",
        "merges_hash": "75a37753dd7a28a2c5df80c28bf06e4e",
    },
}
