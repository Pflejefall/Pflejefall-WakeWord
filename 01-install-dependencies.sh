uv pip install 'git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f'
git clone https://github.com/kahrendt/microWakeWord
uv pip install -e ./microWakeWord
uv pip install torch torchaudio
git clone https://github.com/rhasspy/piper-sample-generator

uv pip install -e ./piper-sample-generator

mv ./microWakeWord/microwakeword ./microwakeword