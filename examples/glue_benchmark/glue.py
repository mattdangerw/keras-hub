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
import csv
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app
from absl import flags
from tensorflow import keras

import keras_nlp

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "task_name",
    "mrpc",
    "The name of the GLUE task to finetune on.",
)

flags.DEFINE_integer(
    "batch_size",
    32,
    "Batch size of data.",
)

flags.DEFINE_integer(
    "epochs",
    2,
    "Number of epochs to run finetuning.",
)

flags.DEFINE_float(
    "learning_rate",
    5e-5,
    "Learning rate",
)

flags.DEFINE_string(
    "submission_directory",
    None,
    "The directory to save the glue submission file.",
)


flags.DEFINE_string(
    "finetuning_model_save_path",
    None,
    "The path to save the finetuning model. If None, the model is not saved.",
)


def load_data(task_name):
    """
    Load GLUE dataset.

    Load GLUE dataset, and convert the dictionary format to (features, label),
    where features is a tuple of all input sentences.
    """
    if task_name in ("cola", "sst2"):
        feature_names = ("sentence",)
    elif task_name in ("mrpc", "stsb", "rte", "wnli"):
        feature_names = ("sentence1", "sentence2")
    elif task_name in ("mnli", "mnli_matched", "mnli_mismatched", "ax"):
        feature_names = ("premise", "hypothesis")
    elif task_name in "qnli":
        feature_names = ("question", "sentence")
    elif task_name in "qqp":
        feature_names = ("question1", "question2")
    else:
        raise ValueError(f"Unknown task_name {task_name}.")

    test_suffix = ""
    if task_name in ("mnli", "mnli_matched"):
        # For "mnli", just run default to "mnli_matched".
        task_name = "mnli"
        test_suffix = "_matched"
    elif task_name in ("mnli_mismatched"):
        task_name = "mnli"
        test_suffix = "_mismatched"

    def split_features(x):
        # GLUE comes with dictonary data, we convert it to a uniform format
        # (features, label), where features is a tuple consisting of all
        # features.
        features = tuple([x[name] for name in feature_names])
        label = x["label"]
        return (features, label)

    if task_name == "ax":
        # AX is trained and evaluated on MNLI, and has its own test split.
        train_ds, validation_ds = tfds.load(
            "glue/mnli",
            split=["train", "validation_matched"],
        )
        test_ds = tfds.load(
            "glue/ax",
            split="test",
        )
    else:
        train_ds, test_ds, validation_ds = tfds.load(
            f"glue/{task_name}",
            split=["train", "test" + test_suffix, "validation" + test_suffix],
        )
    train_ds = train_ds.map(split_features, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(split_features, num_parallel_calls=tf.data.AUTOTUNE)
    validation_ds = validation_ds.map(
        split_features, num_parallel_calls=tf.data.AUTOTUNE
    )
    return train_ds, test_ds, validation_ds


def preprocess_data(preprocess_fn, dataset):
    """Run `proprocess_fn` on input dataset then batch & prefetch."""
    return (
        dataset.map(preprocess_fn)
        .batch(FLAGS.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def generate_submission_files(finetuning_model, test_ds):
    """Generate GLUE leaderboard submission files."""
    filenames = {
        "cola": "CoLA.tsv",
        "sst2": "SST-2.tsv",
        "mrpc": "MRPC.tsv",
        "qqp": "QQP.tsv",
        "stsb": "STS-B.tsv",
        "mnli_matched": "MNLI-m.tsv",
        "mnli_mismatched": "MNLI-mm.tsv",
        "qnli": "QNLI.tsv",
        "rte": "RTE.tsv",
        "wnli": "WNLI.tsv",
        "ax": "AX.tsv",
    }

    labelnames = {
        "mnli_matched": ["entailment", "neutral", "contradiction"],
        "mnli_mismatched": ["entailment", "neutral", "contradiction"],
        "ax": ["entailment", "neutral", "contradiction"],
        "qnli": ["entailment", "not_entailment"],
        "rte": ["entailment", "not_entailment"],
    }
    if not os.path.exists(FLAGS.submission_directory):
        os.makedirs(FLAGS.submission_directory)
    filename = FLAGS.submission_directory + "/" + filenames[FLAGS.task_name]

    labelname = labelnames.get(FLAGS.task_name)
    with tf.io.gfile.GFile(filename, "w") as f:
        # GLUE requires a format of index + tab + prediction.
        writer = csv.writer(f, delimiter="\t")
        # Write the required headline for GLUE.
        writer.writerow(["index", "prediction"])
        predictions = finetuning_model.predict(test_ds)
        predictions = np.argmax(predictions, -1)
        for idx, pred in enumerate(predictions):
            if labelname:
                pred_value = labelname[int(pred)]
            else:
                pred_value = pred
            writer.writerow([idx, pred_value])


def main(_):
    train_ds, test_ds, val_ds = load_data(FLAGS.task_name)

    # ----- Custom code block starts -----
    bert_model = keras_nlp.models.Bert.from_preset("bert_tiny_uncased_en")
    bert_preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
        "bert_tiny_uncased_en"
    )

    # Users should change this function to implement the preprocessing required
    # by the model.
    def preprocess_fn(feature, label):
        return bert_preprocessor(feature), label

    # ----- Custom code block ends -----

    train_ds = preprocess_data(preprocess_fn, train_ds)
    val_ds = preprocess_data(preprocess_fn, val_ds)
    test_ds = preprocess_data(preprocess_fn, test_ds)

    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [keras.metrics.SparseCategoricalAccuracy()]
    if FLAGS.task_name == "stsb":
        num_classes = 1
        loss = keras.losses.MeanSquaredError()
        metrics = [keras.losses.MeanSquaredError()]
    elif FLAGS.task_name in ("mnli", "mnli_mismatched", "mnli_matched", "ax"):
        num_classes = 3
    else:
        num_classes = 2

    # ----- Custom code block starts -----
    # Users should change this `BertClassifier` to your own classifier.
    # Commonly the classifier is simply your model + several dense layers,
    # please refer to "Make the Finetuning Model" section in README for
    # detailed instructions.
    finetuning_model = keras_nlp.models.BertClassifier(
        backbone=bert_model,
        num_classes=num_classes,
    )
    # ----- Custom code block ends -----

    finetuning_model.compile(
        optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate),
        loss=loss,
        metrics=metrics,
    )

    finetuning_model.fit(train_ds, validation_data=val_ds, epochs=FLAGS.epochs)

    if FLAGS.submission_directory:
        generate_submission_files(finetuning_model, test_ds)

    if FLAGS.finetuning_model_save_path:
        finetuning_model.save(FLAGS.finetuning_model_save_path)


if __name__ == "__main__":
    app.run(main)
