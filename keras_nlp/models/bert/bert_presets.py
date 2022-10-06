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
    "bert_tiny_uncased_en": {
        "description": (
            "Tiny size of BERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "backbone_config": {
            "weights": "https://storage.googleapis.com/keras-nlp/models/bert_tiny_uncased_en/model.h5",
            "vocabulary_size": 30522,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 128,
            "intermediate_dim": 512,
            "dropout": 0.1,
            "max_sequence_length": 512,
        },
        "preprocessing_config": {
            "vocabulary": "https://storage.googleapis.com/keras-nlp/models/bert_tiny_uncased_en/vocab.txt",
            "lowercase": True,
        },
    },
    "bert_small_uncased_en": {
        "description": (
            "Small size of BERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "backbone_config": {
            "weights": "https://storage.googleapis.com/keras-nlp/models/bert_small_uncased_en/model.h5",
            "vocabulary_size": 30522,
            "num_layers": 4,
            "num_heads": 8,
            "hidden_dim": 512,
            "intermediate_dim": 2048,
            "dropout": 0.1,
            "max_sequence_length": 512,
        },
        "preprocessing_config": {
            "vocabulary": "https://storage.googleapis.com/keras-nlp/models/bert_small_uncased_en/vocab.txt",
            "lowercase": True,
        },
    },
    "bert_medium_uncased_en": {
        "description": (
            "Medium size of BERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "backbone_config": {
            "weights": "https://storage.googleapis.com/keras-nlp/models/bert_medium_uncased_en/model.h5",
            "vocabulary_size": 30522,
            "num_layers": 8,
            "num_heads": 8,
            "hidden_dim": 512,
            "intermediate_dim": 2048,
            "dropout": 0.1,
            "max_sequence_length": 512,
        },
        "preprocessing_config": {
            "vocabulary": "https://storage.googleapis.com/keras-nlp/models/bert_medium_uncased_en/vocab.txt",
            "lowercase": True,
        },
    },
    "bert_base_uncased_en": {
        "description": (
            "Base size of BERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "backbone_config": {
            "weights": "https://storage.googleapis.com/keras-nlp/models/bert_base_uncased_en/model.h5",
            "vocabulary_size": 30522,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "max_sequence_length": 512,
        },
        "preprocessing_config": {
            "vocabulary": "https://storage.googleapis.com/keras-nlp/models/bert_base_uncased_en/vocab.txt",
            "lowercase": True,
        },
    },
    "bert_base_cased_en": {
        "description": (
            "Base size of BERT where case is maintained. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "backbone_config": {
            "weights": "https://storage.googleapis.com/keras-nlp/models/bert_base_cased_en/model.h5",
            "vocabulary_size": 28996,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "max_sequence_length": 512,
        },
        "preprocessing_config": {
            "vocabulary": "https://storage.googleapis.com/keras-nlp/models/bert_base_cased_en/vocab.txt",
            "lowercase": False,
        },
    },
    "bert_base_zh": {
        "description": ("Base size of BERT. Trained on Chinese Wikipedia."),
        "backbone_config": {
            "weights": "https://storage.googleapis.com/keras-nlp/models/bert_base_zh/model.h5",
            "vocabulary_size": 21128,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "max_sequence_length": 512,
        },
        "preprocessing_config": {
            "vocabulary": "https://storage.googleapis.com/keras-nlp/models/bert_base_zh/vocab.txt",
            "lowercase": False,
        },
    },
    "bert_base_multi_cased": {
        "description": ("Base size of BERT. Trained on Wikipedia."),
        "backbone_config": {
            "weights": "https://storage.googleapis.com/keras-nlp/models/bert_base_multi_cased/model.h5",
            "vocabulary_size": 119547,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "max_sequence_length": 512,
        },
        "preprocessing_config": {
            "vocabulary": "https://storage.googleapis.com/keras-nlp/models/bert_base_multi_cased/vocab.txt",
            "lowercase": False,
        },
    },
    "bert_large_uncased_en": {
        "description": (
            "Large size of BERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "backbone_config": {
            "weights": "https://storage.googleapis.com/keras-nlp/models/bert_large_uncased_en/model.h5",
            "vocabulary_size": 30522,
            "num_layers": 24,
            "num_heads": 16,
            "hidden_dim": 1024,
            "intermediate_dim": 4096,
            "dropout": 0.1,
            "max_sequence_length": 512,
        },
        "preprocessing_config": {
            "vocabulary": "https://storage.googleapis.com/keras-nlp/models/bert_large_uncased_en/vocab.txt",
            "lowercase": True,
        },
    },
    "bert_large_cased_en": {
        "description": (
            "Base size of BERT where case is maintained. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "backbone_config": {
            "weights": "https://storage.googleapis.com/keras-nlp/models/bert_large_cased_en/model.h5",
            "vocabulary_size": 28996,
            "num_layers": 24,
            "num_heads": 16,
            "hidden_dim": 1024,
            "intermediate_dim": 4096,
            "dropout": 0.1,
            "max_sequence_length": 512,
        },
        "preprocessing_config": {
            "vocabulary": "https://storage.googleapis.com/keras-nlp/models/bert_base_cased_en/vocab.txt",
            "lowercase": False,
        },
    },
}

classifier_presets = {
    "bert_base_uncased_en_sst2": {
        "description": (
            "Base size of BERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus, fine tuned on sst2."
        ),
        "classifier_config": {
            "weights": None,  # TODO(mattdangerw): we don't have this yet.
            "backbone": {
                "class_name": "keras_nlp>Bert",  # This is how core Keras recognizes a package symbol.
                "config": {
                    "vocabulary_size": 28996,
                    "num_layers": 24,
                    "num_heads": 16,
                    "hidden_dim": 1024,
                    "intermediate_dim": 4096,
                    "dropout": 0.1,
                    "max_sequence_length": 512,
                },
            },
            "num_classes": 2,
        },
        "preprocessing_config": {
            "vocabulary": "https://storage.googleapis.com/keras-nlp/models/bert_base_uncased_en/vocab.txt",
            "lowercase": True,
        },
    },
}

all_presets = {**backbone_presets, **classifier_presets}


def pluck(dict, inner_key):
    return {k: v[inner_key] for k, v in dict.items()}


def invert(dict):
    return {v: k for k, v in dict.items()}


preprocessing_configs = pluck(all_presets, "preprocessing_config")
vocab_id_to_url = pluck(preprocessing_configs, "vocabulary")
vocab_url_to_id = invert(vocab_id_to_url)

backbone_configs = pluck(backbone_presets, "backbone_config")
backbone_weight_id_to_url = pluck(backbone_configs, "weights")
backbone_weight_url_to_id = invert(backbone_weight_id_to_url)

classifier_configs = pluck(classifier_presets, "classifier_config")
classifier_weight_id_to_url = pluck(classifier_configs, "weights")
classifier_weight_url_to_id = invert(classifier_weight_id_to_url)
