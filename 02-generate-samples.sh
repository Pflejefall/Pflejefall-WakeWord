python3 \
    ./piper-sample-generator/generate_samples.py \
    'pflejefall.' \
    --model piper_voices/de_DE-thorsten-medium.onnx \
    --max-samples 500 \
    --batch-size 10 \
    --output-dir generated_samples/