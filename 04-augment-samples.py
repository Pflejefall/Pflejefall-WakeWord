# Augment samples and save the training, validation, and testing sets.
# Validating and testing samples generated the same way can make the model
# benchmark better than it performs in real-word use. Use real samples or TTS
# samples generated with a different TTS engine to potentially get more accurate
# benchmarks.

import os
from mmap_ninja.ragged import RaggedMmap
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration

output_dir = 'generated_augmented_features'

clips = Clips(input_directory='generated_samples',
    file_pattern='*.wav',
    max_clip_duration_s=None,
    remove_silence=False,
    random_split_seed=10,
    split_count=0.1,
)

augmenter = Augmentation(augmentation_duration_s=3.2,
    augmentation_probabilities = {
        "SevenBandParametricEQ": 0.1,
        "TanhDistortion": 0.1,
        "PitchShift": 0.1,
        "BandStopFilter": 0.1,
        "AddColorNoise": 0.1,
        "AddBackgroundNoise": 0.75,
        "Gain": 1.0,
        "RIR": 0.5,
    },
    impulse_paths = ['mit_rirs'],
    background_paths = ['fma_16k', 'audioset_16k'],
    background_min_snr_db = -5,
    background_max_snr_db = 10,
    min_jitter_s = 0.195,
    max_jitter_s = 0.205,
)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

splits = ["training", "validation", "testing"]
for split in splits:
  out_dir = os.path.join(output_dir, split)
  if not os.path.exists(out_dir):
      os.mkdir(out_dir)


  split_name = "train"
  repetition = 2

  spectrograms = SpectrogramGeneration(clips=clips,
                                     augmenter=augmenter,
                                     slide_frames=10,    # Uses the same spectrogram repeatedly, just shifted over by one frame. This simulates the streaming inferences while training/validating in nonstreaming mode.
                                     step_ms=10,
                                     )
  if split == "validation":
    split_name = "validation"
    repetition = 1
  elif split == "testing":
    split_name = "test"
    repetition = 1
    spectrograms = SpectrogramGeneration(clips=clips,
                                     augmenter=augmenter,
                                     slide_frames=1,    # The testing set uses the streaming version of the model, so no artificial repetition is necessary
                                     step_ms=10,
                                     )

  RaggedMmap.from_generator(
      out_dir=os.path.join(out_dir, 'wakeword_mmap'),
      sample_generator=spectrograms.spectrogram_generator(split=split_name, repeat=repetition),
      batch_size=100,
      verbose=True,
  )