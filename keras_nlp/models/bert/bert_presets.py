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
"""BERT model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "bert_tiny_en_uncased": {
        "backbone": {
            "class_name": "keras_nlp>BertBackbone",
            "vocabulary_size": 30522,
            "hidden_dim": 128,
            "intermediate_dim": 512,
            "num_layers": 2,
            "num_heads": 2,
            "max_sequence_length": 512,
            "num_segments": 2,
            "dropout": 0.1,
            "weights": {
                "filename": "model.h5",
                "url": "https://storage.googleapis.com/keras-nlp/models/bert_tiny_en_uncased/v1/model.h5",
                "hash": "c2b29fcbf8f814a0812e4ab89ef5c068",
            },
        },
        "preprocessor": {
            "class_name": "keras_nlp>BertPreprocessor",
            "tokenizer": {
                "class_name": "keras_nlp>BertTokenizer",
                "vocabulary": {
                    "filename": "vocab.txt",
                    "url": "https://storage.googleapis.com/keras-nlp/models/bert_tiny_en_uncased/v1/vocab.txt",
                    "hash": "64800d5d8528ce344256daf115d4965e",
                },
                "lowercase": True,
            },
            "sequence_length": 512,
        },
    },
    "bert_small_en_uncased": {
        "backbone": {
            "class_name": "keras_nlp>BertBackbone",
            "vocabulary_size": 30522,
            "hidden_dim": 512,
            "intermediate_dim": 2048,
            "num_layers": 4,
            "num_heads": 8,
            "max_sequence_length": 512,
            "num_segments": 2,
            "dropout": 0.1,
            "weights": {
                "filename": "model.h5",
                "url": "https://storage.googleapis.com/keras-nlp/models/bert_small_en_uncased/v1/model.h5",
                "hash": "08632c9479b034f342ba2c2b7afba5f7",
            },
        },
        "preprocessor": {
            "class_name": "keras_nlp>BertPreprocessor",
            "tokenizer": {
                "class_name": "keras_nlp>BertTokenizer",
                "vocabulary": {
                    "filename": "vocab.txt",
                    "url": "https://storage.googleapis.com/keras-nlp/models/bert_small_en_uncased/v1/vocab.txt",
                    "hash": "64800d5d8528ce344256daf115d4965e",
                },
                "lowercase": True,
            },
            "sequence_length": 512,
        },
    },
    "bert_medium_en_uncased": {
        "backbone": {
            "class_name": "keras_nlp>BertBackbone",
            "vocabulary_size": 30522,
            "hidden_dim": 512,
            "intermediate_dim": 2048,
            "num_layers": 8,
            "num_heads": 8,
            "max_sequence_length": 512,
            "num_segments": 2,
            "dropout": 0.1,
            "weights": {
                "filename": "model.h5",
                "url": "https://storage.googleapis.com/keras-nlp/models/bert_medium_en_uncased/v1/model.h5",
                "hash": "bb990e1184ec6b6185450c73833cd661",
            },
        },
        "preprocessor": {
            "class_name": "keras_nlp>BertPreprocessor",
            "tokenizer": {
                "class_name": "keras_nlp>BertTokenizer",
                "vocabulary": {
                    "filename": "vocab.txt",
                    "url": "https://storage.googleapis.com/keras-nlp/models/bert_medium_en_uncased/v1/vocab.txt",
                    "hash": "64800d5d8528ce344256daf115d4965e",
                },
                "lowercase": True,
            },
            "sequence_length": 512,
        },
    },
    "bert_base_en_uncased": {
        "backbone": {
            "class_name": "keras_nlp>BertBackbone",
            "vocabulary_size": 30522,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "num_layers": 12,
            "num_heads": 12,
            "max_sequence_length": 512,
            "num_segments": 2,
            "dropout": 0.1,
            "weights": {
                "filename": "model.h5",
                "url": "https://storage.googleapis.com/keras-nlp/models/bert_base_en_uncased/v1/model.h5",
                "hash": "9b2b2139f221988759ac9cdd17050b31",
            },
        },
        "preprocessor": {
            "class_name": "keras_nlp>BertPreprocessor",
            "tokenizer": {
                "class_name": "keras_nlp>BertTokenizer",
                "vocabulary": {
                    "filename": "vocab.txt",
                    "url": "https://storage.googleapis.com/keras-nlp/models/bert_base_en_uncased/v1/vocab.txt",
                    "hash": "64800d5d8528ce344256daf115d4965e",
                },
                "lowercase": True,
            },
            "sequence_length": 512,
        },
    },
    "bert_base_en": {
        "backbone": {
            "class_name": "keras_nlp>BertBackbone",
            "vocabulary_size": 28996,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "num_layers": 12,
            "num_heads": 12,
            "max_sequence_length": 512,
            "num_segments": 2,
            "dropout": 0.1,
            "weights": {
                "filename": "model.h5",
                "url": "https://storage.googleapis.com/keras-nlp/models/bert_base_en/v1/model.h5",
                "hash": "f94a6cb012e18f4fb8ec92abb91864e9",
            },
        },
        "preprocessor": {
            "class_name": "keras_nlp>BertPreprocessor",
            "tokenizer": {
                "class_name": "keras_nlp>BertTokenizer",
                "vocabulary": {
                    "filename": "vocab.txt",
                    "url": "https://storage.googleapis.com/keras-nlp/models/bert_base_en/v1/vocab.txt",
                    "hash": "bb6ca9b42e790e5cd986bbb16444d0e0",
                },
                "lowercase": False,
            },
            "sequence_length": 512,
        },
    },
    "bert_base_zh": {
        "backbone": {
            "class_name": "keras_nlp>BertBackbone",
            "vocabulary_size": 21128,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "num_layers": 12,
            "num_heads": 12,
            "max_sequence_length": 512,
            "num_segments": 2,
            "dropout": 0.1,
            "weights": {
                "filename": "model.h5",
                "url": "https://storage.googleapis.com/keras-nlp/models/bert_base_zh/v1/model.h5",
                "hash": "79afa421e386076e62ab42dad555ab0c",
            },
        },
        "preprocessor": {
            "class_name": "keras_nlp>BertPreprocessor",
            "tokenizer": {
                "class_name": "keras_nlp>BertTokenizer",
                "vocabulary": {
                    "filename": "vocab.txt",
                    "url": "https://storage.googleapis.com/keras-nlp/models/bert_base_zh/v1/vocab.txt",
                    "hash": "3b5b76c4aef48ecf8cb3abaafe960f09",
                },
                "lowercase": False,
            },
            "sequence_length": 512,
        },
    },
    "bert_base_multi": {
        "backbone": {
            "class_name": "keras_nlp>BertBackbone",
            "vocabulary_size": 119547,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "num_layers": 12,
            "num_heads": 12,
            "max_sequence_length": 512,
            "num_segments": 2,
            "dropout": 0.1,
            "weights": {
                "filename": "model.h5",
                "url": "https://storage.googleapis.com/keras-nlp/models/bert_base_multi/v1/model.h5",
                "hash": "b0631cec0a1f2513c6cfd75ba29c33aa",
            },
        },
        "preprocessor": {
            "class_name": "keras_nlp>BertPreprocessor",
            "tokenizer": {
                "class_name": "keras_nlp>BertTokenizer",
                "vocabulary": {
                    "filename": "vocab.txt",
                    "url": "https://storage.googleapis.com/keras-nlp/models/bert_base_multi/v1/vocab.txt",
                    "hash": "d9d865138d17f1958502ed060ecfeeb6",
                },
                "lowercase": False,
            },
            "sequence_length": 512,
        },
    },
    "bert_large_en_uncased": {
        "backbone": {
            "class_name": "keras_nlp>BertBackbone",
            "vocabulary_size": 30522,
            "hidden_dim": 1024,
            "intermediate_dim": 4096,
            "num_layers": 24,
            "num_heads": 16,
            "max_sequence_length": 512,
            "num_segments": 2,
            "dropout": 0.1,
            "weights": {
                "filename": "model.h5",
                "url": "https://storage.googleapis.com/keras-nlp/models/bert_large_en_uncased/v1/model.h5",
                "hash": "cc5cacc9565ef400ee4376105f40ddae",
            },
        },
        "preprocessor": {
            "class_name": "keras_nlp>BertPreprocessor",
            "tokenizer": {
                "class_name": "keras_nlp>BertTokenizer",
                "vocabulary": {
                    "filename": "vocab.txt",
                    "url": "https://storage.googleapis.com/keras-nlp/models/bert_large_en_uncased/v1/vocab.txt",
                    "hash": "64800d5d8528ce344256daf115d4965e",
                },
                "lowercase": True,
            },
            "sequence_length": 512,
        },
    },
    "bert_large_en": {
        "backbone": {
            "class_name": "keras_nlp>BertBackbone",
            "vocabulary_size": 28996,
            "hidden_dim": 1024,
            "intermediate_dim": 4096,
            "num_layers": 24,
            "num_heads": 16,
            "max_sequence_length": 512,
            "num_segments": 2,
            "dropout": 0.1,
            "weights": {
                "filename": "model.h5",
                "url": "https://storage.googleapis.com/keras-nlp/models/bert_large_en/v1/model.h5",
                "hash": "8b8ab82290bbf4f8db87d4f100648890",
            },
        },
        "preprocessor": {
            "class_name": "keras_nlp>BertPreprocessor",
            "tokenizer": {
                "class_name": "keras_nlp>BertTokenizer",
                "vocabulary": {
                    "filename": "vocab.txt",
                    "url": "https://storage.googleapis.com/keras-nlp/models/bert_large_en/v1/vocab.txt",
                    "hash": "bb6ca9b42e790e5cd986bbb16444d0e0",
                },
                "lowercase": False,
            },
            "sequence_length": 512,
        },
    },
}

classifier_presets = {
    "bert_tiny_en_uncased_sst2": {
        "class_name": "keras_nlp>BertClassifier",
        "backbone": {
            "class_name": "keras_nlp>BertBackbone",
            "vocabulary_size": 30522,
            "hidden_dim": 128,
            "intermediate_dim": 512,
            "num_layers": 2,
            "num_heads": 2,
            "max_sequence_length": 512,
            "num_segments": 2,
            "dropout": 0.1,
        },
        "preprocessor": {
            "class_name": "keras_nlp>BertPreprocessor",
            "tokenizer": {
                "class_name": "keras_nlp>BertTokenizer",
                "vocabulary": {
                    "filename": "vocab.txt",
                    "url": "https://storage.googleapis.com/keras-nlp/models/bert_tiny_en_uncased_sst2/v1/vocab.txt",
                    "hash": "64800d5d8528ce344256daf115d4965e",
                },
            },
            "sequence_length": 512,
        },
        "num_classes": 2,
        "dropout": 0.1,
        "weights": {
            "filename": "model.h5",
            "url": "https://storage.googleapis.com/keras-nlp/models/bert_tiny_en_uncased_sst2/v1/model.h5",
            "hash": "1f9c2d59f9e229e08f3fbd44239cfb0b",
        },
    },
}
