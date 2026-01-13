# Trains a model. When finished, it will quantize and convert the model to a
# streaming version suitable for on-device detection.
# It will resume if stopped, but it will start over at the configured training
# steps in the yaml file.
# Change --train 0 to only convert and test the best-weighted model.
# On Google colab, it doesn't print the mini-batch results, so it may appear
# stuck for several minutes! Additionally, it is very slow compared to training
# on a local GPU.

# Disable GPU usage
export CUDA_VISIBLE_DEVICES=""

export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_GPU_MAX_ALLOC_SIZE=536870912  # 512MB

python -m microwakeword.model_train_eval \
    --training_config='training_parameters.yaml' \
    --train 1 \
    --restore_checkpoint 1 \
    --test_tf_nonstreaming 0 \
    --test_tflite_nonstreaming 0 \
    --test_tflite_nonstreaming_quantized 0 \
    --test_tflite_streaming 0 \
    --test_tflite_streaming_quantized 1 \
    --use_weights "best_weights" \
    mixednet \
    --pointwise_filters "64,64,64,64" \
    --repeat_in_block  "1, 1, 1, 1" \
    --mixconv_kernel_sizes "[3], [3,5], [3,5], [5]" \
    --residual_connection "0,0,0,0" \
    --first_conv_filters 32 \
    --first_conv_kernel_size 5 \
    --stride 3
