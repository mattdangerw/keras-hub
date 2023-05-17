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
import keras_core
import tensorflow as tf

from keras_nlp.backend.config import multi_backend

if multi_backend():
    from keras_core.src.operations import *  # noqa: F403, F401
else:
    from keras_core.src.backend.tensorflow import *  # noqa: F403, F401
    from keras_core.src.backend.tensorflow.core import *  # noqa: F403, F401
    from keras_core.src.backend.tensorflow.math import *  # noqa: F403, F401
    from keras_core.src.backend.tensorflow.nn import *  # noqa: F403, F401
    from keras_core.src.backend.tensorflow.numpy import *  # noqa: F403, F401


# Override arange, the tf numpy does not work with placeholder tensors.
if not multi_backend or keras_core.config.backend() == "tensorflow":

    def arange(start, stop=None, step=1, dtype=None):
        return tf.range(start, stop, delta=step, dtype=dtype)
