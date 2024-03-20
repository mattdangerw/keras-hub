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

backbone_presets = {
    "bert_tiny_en_uncased": "kaggle://keras/bert/keras/bert_tiny_en_uncased/2",
    "bert_small_en_uncased": "kaggle://keras/bert/keras/bert_small_en_uncased/2",
    "bert_medium_en_uncased":  "kaggle://keras/bert/keras/bert_medium_en_uncased/2",
    "bert_base_en_uncased": "kaggle://keras/bert/keras/bert_base_en_uncased/2",
    "bert_base_en": "kaggle://keras/bert/keras/bert_base_en/2",
    "bert_base_zh": "kaggle://keras/bert/keras/bert_base_zh/2",
    "bert_base_multi": "kaggle://keras/bert/keras/bert_base_multi/2",
    "bert_large_en_uncased": "kaggle://keras/bert/keras/bert_large_en_uncased/2",
    "bert_large_en": "kaggle://keras/bert/keras/bert_large_en/2",
}

classifier_presets = {
    "bert_tiny_en_uncased_sst2": "kaggle://keras/bert/keras/bert_tiny_en_uncased_sst2/3",
}
