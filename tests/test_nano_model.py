#!/usr/bin/env python3
"""
Test script for YOLO11 nano TRT model in Triton.
This tests the raw model inference without NMS post-processing.
"""

import sys

import numpy as np
import tritonclient.http as httpclient
from PIL import Image


def preprocess_image(image_path, input_shape=(640, 640)):
    """Preprocess image for YOLO inference."""
    # Load image
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (width, height)

    # Resize
    img = img.resize(input_shape)

    # Convert to numpy array and normalize
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0  # Normalize to [0, 1]

    # Transpose to CHW format
    img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)  # (1, 3, 640, 640)

    # Convert to FP16 for the model
    img_array = img_array.astype(np.float16)

    return img_array, original_size


def test_inference(image_path, model_name='yolov11_nano_trt'):
    """Test inference with Triton model."""
    print(f'Testing model: {model_name}')
    print(f'Image: {image_path}')

    # Create Triton client
    triton_client = httpclient.InferenceServerClient(url='localhost:4600')

    # Check if model is ready
    if not triton_client.is_model_ready(model_name):
        print(f'ERROR: Model {model_name} is not ready!')
        return False

    print(f'✓ Model {model_name} is ready')

    # Get model metadata
    metadata = triton_client.get_model_metadata(model_name)
    print('✓ Model metadata retrieved')
    print(f'  Inputs: {[inp["name"] for inp in metadata["inputs"]]}')
    print(f'  Outputs: {[out["name"] for out in metadata["outputs"]]}')

    # Preprocess image
    input_data, _original_size = preprocess_image(image_path)
    print(f'✓ Image preprocessed: {input_data.shape}, dtype={input_data.dtype}')

    # Prepare input
    inputs = [httpclient.InferInput('images', input_data.shape, 'FP16')]
    inputs[0].set_data_from_numpy(input_data)

    # Prepare output
    outputs = [httpclient.InferRequestedOutput('output0')]

    # Run inference
    print('Running inference...')
    response = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

    # Get output
    output = response.as_numpy('output0')
    print('✓ Inference successful!')
    print(f'  Output shape: {output.shape}')
    print(f'  Output dtype: {output.dtype}')
    print(f'  Output range: [{output.min():.4f}, {output.max():.4f}]')

    # Parse detections (simple check)
    # Output shape is (1, 84, 8400) - 84 = 4 bbox coords + 80 classes
    # Each of 8400 anchor points has 84 values
    batch_size, num_features, num_anchors = output.shape
    print('\nModel output analysis:')
    print(f'  Batch size: {batch_size}')
    print(f'  Features per anchor: {num_features} (4 bbox + 80 classes)')
    print(f'  Anchor points: {num_anchors}')

    # Count potential detections (naive threshold)
    # Get max class score for each anchor
    class_scores = output[0, 4:, :]  # (80, 8400)
    max_scores = class_scores.max(axis=0)  # (8400,)

    # Count how many have confidence > 0.5
    high_conf = (max_scores > 0.5).sum()
    print('\nDetection stats (threshold=0.5):')
    print(f'  Anchors with high confidence: {high_conf}')
    print(f'  Max confidence score: {max_scores.max():.4f}')

    if high_conf > 0:
        print('\n✓ Model appears to be detecting objects!')
    else:
        print('\n⚠ No high-confidence detections (may need NMS post-processing)')

    return True


if __name__ == '__main__':
    image_path = sys.argv[1] if len(sys.argv) > 1 else '/app/test_images/bus.jpg'

    try:
        success = test_inference(image_path)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f'\n✗ Error: {e}')
        import traceback

        traceback.print_exc()
        sys.exit(1)
