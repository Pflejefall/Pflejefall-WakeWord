python3 \
    ./piper-sample-generator/generate_samples.py \
    'pfleyefall?' \
    --model piper_voices/de_DE-thorsten-medium.onnx \
    --max-samples 1000 \
    --batch-size 10 \
    --output-dir generated_samples/