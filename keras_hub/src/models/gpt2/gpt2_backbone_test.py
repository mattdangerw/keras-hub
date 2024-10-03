import pytest
from keras import ops

from keras_hub.src.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_hub.src.tests.test_case import TestCase


class GPT2BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 2,
            "intermediate_dim": 4,
            "max_sequence_length": 5,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=GPT2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 2),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=GPT2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=GPT2Backbone,
            preset="gpt2_base_en",
            input_data={
                "token_ids": ops.array([[1169, 2068, 7586, 21831, 13]]),
                "padding_mask": ops.ones((1, 5), dtype="int32"),
            },
            expected_output_shape=(1, 5, 768),
            # The forward pass from a preset should be stable!
            expected_partial_output=ops.array(
                [-0.1116, -0.0375, -0.2624, 0.00891, -0.0061]
            ),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in GPT2Backbone.presets:
            self.run_preset_test(
                cls=GPT2Backbone,
                preset=preset,
                input_data=self.input_data,
            )