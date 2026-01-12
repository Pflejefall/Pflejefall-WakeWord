# Save a yaml config that controls the training process
# These hyperparamters can make a huge different in model quality.
# Experiment with sampling and penalty weights and increasing the number of
# training steps.

import yaml
import os

config = {}

config["window_step_ms"] = 10

config["train_dir"] = (
    "trained_models/wakeword"
)


# Each feature_dir should have at least one of the following folders with this structure:
#  training/
#    ragged_mmap_folders_ending_in_mmap
#  testing/
#    ragged_mmap_folders_ending_in_mmap
#  testing_ambient/
#    ragged_mmap_folders_ending_in_mmap
#  validation/
#    ragged_mmap_folders_ending_in_mmap
#  validation_ambient/
#    ragged_mmap_folders_ending_in_mmap
#
#  sampling_weight: Weight for choosing a spectrogram from this set in the batch
#  penalty_weight: Penalizing weight for incorrect predictions from this set
#  truth: Boolean whether this set has positive samples or negative samples
#  truncation_strategy = If spectrograms in the set are longer than necessary for training, how are they truncated
#       - random: choose a random portion of the entire spectrogram - useful for long negative samples
#       - truncate_start: remove the start of the spectrogram
#       - truncate_end: remove the end of the spectrogram
#       - split: Split the longer spectrogram into separate spectrograms offset by 100 ms. Only for ambient sets

config["features"] = [
    {
        "features_dir": "generated_augmented_features",
        "sampling_weight": 2.0,
        "penalty_weight": 1.0,
        "truth": True,
        "truncation_strategy": "truncate_start",
        "type": "mmap",
    },
    {
        "features_dir": "negative_datasets/speech",
        "sampling_weight": 10.0,
        "penalty_weight": 1.0,
        "truth": False,
        "truncation_strategy": "random",
        "type": "mmap",
    },
    {
        "features_dir": "negative_datasets/dinner_party",
        "sampling_weight": 10.0,
        "penalty_weight": 1.0,
        "truth": False,
        "truncation_strategy": "random",
        "type": "mmap",
    },
    {
        "features_dir": "negative_datasets/no_speech",
        "sampling_weight": 5.0,
        "penalty_weight": 1.0,
        "truth": False,
        "truncation_strategy": "random",
        "type": "mmap",
    },
    { # Only used for validation and testing
        "features_dir": "negative_datasets/dinner_party_eval",
        "sampling_weight": 0.0,
        "penalty_weight": 1.0,
        "truth": False,
        "truncation_strategy": "split",
        "type": "mmap",
    },
]

# Number of training steps in each iteration - various other settings are configured as lists that corresponds to different steps
config["training_steps"] = [10000]

# Penalizing weight for incorrect class predictions - lists that correspond to training steps
config["positive_class_weight"] = [1]
config["negative_class_weight"] = [20]

config["learning_rates"] = [
    0.001,
]  # Learning rates for Adam optimizer - list that corresponds to training steps
config["batch_size"] = 128

config["time_mask_max_size"] = [
    0
]  # SpecAugment - list that corresponds to training steps
config["time_mask_count"] = [0]  # SpecAugment - list that corresponds to training steps
config["freq_mask_max_size"] = [
    0
]  # SpecAugment - list that corresponds to training steps
config["freq_mask_count"] = [0]  # SpecAugment - list that corresponds to training steps

config["eval_step_interval"] = (
    500  # Test the validation sets after every this many steps
)
config["clip_duration_ms"] = (
    1500  # Maximum length of wake word that the streaming model will accept
)

# The best model weights are chosen first by minimizing the specified minimization metric below the specified target_minimization
# Once the target has been met, it chooses the maximum of the maximization metric. Set 'minimization_metric' to None to only maximize
# Available metrics:
#   - "loss" - cross entropy error on validation set
#   - "accuracy" - accuracy of validation set
#   - "recall" - recall of validation set
#   - "precision" - precision of validation set
#   - "false_positive_rate" - false positive rate of validation set
#   - "false_negative_rate" - false negative rate of validation set
#   - "ambient_false_positives" - count of false positives from the split validation_ambient set
#   - "ambient_false_positives_per_hour" - estimated number of false positives per hour on the split validation_ambient set
config["target_minimization"] = 0.9
config["minimization_metric"] = None  # Set to None to disable

config["maximization_metric"] = "average_viable_recall"

with open(os.path.join("training_parameters.yaml"), "w") as file:
    documents = yaml.dump(config, file)