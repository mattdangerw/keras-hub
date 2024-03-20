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
"""GPT-2 model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "gpt2_base_en": "kaggle://keras/gpt2/keras/gpt2_base_en/2",
    "gpt2_medium_en": "kaggle://keras/gpt2/keras/gpt2_medium_en/2",
    "gpt2_large_en": "kaggle://keras/gpt2/keras/gpt2_large_en/2",
    "gpt2_extra_large_en": "kaggle://keras/gpt2/keras/gpt2_extra_large_en/2",
    "gpt2_base_en_cnn_dailymail": "kaggle://keras/gpt2/keras/gpt2_base_en_cnn_dailymail/2",
}
