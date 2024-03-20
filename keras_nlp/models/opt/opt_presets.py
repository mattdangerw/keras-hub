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
"""OPT model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "opt_125m_en": "kaggle://keras/opt/keras/opt_125m_en/2",
    # We skip the 350m checkpoint because it does not match the structure of
    # other checkpoints.
    "opt_1.3b_en": "kaggle://keras/opt/keras/opt_1.3b_en/2",
    "opt_2.7b_en": "kaggle://keras/opt/keras/opt_2.7b_en/2",
    "opt_6.7b_en": "kaggle://keras/opt/keras/opt_6.7b_en/2",
}
