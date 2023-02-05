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
"""RoBERTa model preset configurations."""

backbone_presets = {
    "roberta_base_en": {
        "backbone": {
            "class_name": "keras_nlp>RobertaBackbone",
            "vocabulary_size": 50265,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "max_sequence_length": 512,
            "weights": {
                "filename": "model.h5",
                "url": "https://storage.googleapis.com/keras-nlp/models/roberta_base_en/v1/model.h5",
                "hash": "958eede1c7edaa9308e027be18fde7a8",
            },
        },
        "preprocessor": {
            "class_name": "keras_nlp>RobertaPreprocessor",
            "tokenizer": {
                "class_name": "keras_nlp>RobertaTokenizer",
                "vocabulary": {
                    "filename": "vocab.json",
                    "url": "https://storage.googleapis.com/keras-nlp/models/roberta_base_en/v1/vocab.json",
                    "hash": "be4d3c6f3f5495426b2c03b334334354",
                },
                "merges": {
                    "filename": "merges.txt",
                    "url": "https://storage.googleapis.com/keras-nlp/models/roberta_base_en/v1/merges.txt",
                    "hash": "75a37753dd7a28a2c5df80c28bf06e4e",
                },
            },
            "sequence_length": 512,
        },
    },
    "roberta_large_en": {
        "backbone": {
            "class_name": "keras_nlp>RobertaBackbone",
            "vocabulary_size": 50265,
            "num_layers": 24,
            "num_heads": 16,
            "hidden_dim": 1024,
            "intermediate_dim": 4096,
            "dropout": 0.1,
            "max_sequence_length": 512,
            "weights": {
                "filename": "model.h5",
                "url": "https://storage.googleapis.com/keras-nlp/models/roberta_large_en/v1/model.h5",
                "hash": "1978b864c317a697fe62a894d3664f14",
            },
        },
        "preprocessor": {
            "class_name": "keras_nlp>RobertaPreprocessor",
            "tokenizer": {
                "class_name": "keras_nlp>RobertaTokenizer",
                "vocabulary": {
                    "filename": "vocab.json",
                    "url": "https://storage.googleapis.com/keras-nlp/models/roberta_large_en/v1/vocab.json",
                    "hash": "be4d3c6f3f5495426b2c03b334334354",
                },
                "merges": {
                    "filename": "merges.txt",
                    "url": "https://storage.googleapis.com/keras-nlp/models/roberta_large_en/v1/merges.txt",
                    "hash": "75a37753dd7a28a2c5df80c28bf06e4e",
                },
            },
            "sequence_length": 512,
        },
    },
}
