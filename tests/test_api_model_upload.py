#!/usr/bin/env python3
"""
Test API-based model upload workflow.

This script validates the complete user workflow:
1. Upload YOLO11 model via API
2. Monitor export progress
3. Verify Triton loading
4. Test inference through normal API endpoints (/detect, /analyze)
5. Clean up test model
"""

import asyncio
import time
from pathlib import Path

import httpx


# Configuration
API_BASE = 'http://localhost:4603'
TRITON_BASE = 'http://localhost:4600'
MODEL_PATH = Path('pytorch_models/yolo11s.pt')
TEST_IMAGE = Path('test_images/zidane.jpg')
TRITON_NAME = 'yolo11s_test_upload'
TIMEOUT = 600  # 10 minutes for export (TRT End2End build takes 6-7 min)


async def test_upload_workflow():  # noqa: PLR0911
    """Test complete model upload workflow."""
    print('=' * 80)
    print('API-Based Model Upload Workflow Test')
    print('=' * 80)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Upload model
        print('\n[1/7] Uploading model to API...')
        print(f'  Model: {MODEL_PATH}')
        print(f'  Target name: {TRITON_NAME}')

        if not MODEL_PATH.exists():
            print(f'❌ Model file not found: {MODEL_PATH}')
            return False

        with open(MODEL_PATH, 'rb') as f:
            files = {'file': (MODEL_PATH.name, f, 'application/octet-stream')}
            data = {
                'triton_name': TRITON_NAME,
                'max_batch': '32',
                'formats': 'trt_end2end',
                'auto_load': 'true',
            }

            response = await client.post(
                f'{API_BASE}/models/upload',
                files=files,
                data=data,
            )

        if response.status_code != 200:
            print(f'❌ Upload failed: {response.status_code}')
            print(f'  Response: {response.text}')
            return False

        result = response.json()
        task_id = result['task_id']
        print('✅ Upload successful!')
        print(f'  Task ID: {task_id}')
        print(f'  Status: {result["status"]}')

        # Step 2: Monitor export progress
        print('\n[2/7] Monitoring export progress...')
        start_time = time.time()
        last_status = None

        while True:
            if time.time() - start_time > TIMEOUT:
                print(f'❌ Export timeout after {TIMEOUT}s')
                return False

            response = await client.get(f'{API_BASE}/models/export/{task_id}')
            if response.status_code != 200:
                print(f'❌ Status check failed: {response.status_code}')
                return False

            status = response.json()

            # Print status updates
            if status['status'] != last_status:
                elapsed = time.time() - start_time
                print(f'  [{elapsed:.1f}s] Status: {status["status"]}')
                if status.get('message'):
                    print(f'    Message: {status["message"]}')
                if status.get('progress'):
                    print(f'    Progress: {status["progress"]}')
                last_status = status['status']

            # Check completion
            if status['status'] == 'completed':
                print(f'✅ Export completed in {time.time() - start_time:.1f}s')
                if status.get('output_path'):
                    print(f'  Model path: {status["output_path"]}')
                break
            if status['status'] == 'failed':
                print('❌ Export failed!')
                print(f'  Error: {status.get("error", "Unknown error")}')
                return False

            await asyncio.sleep(2)

        # Step 3: Verify Triton loading
        print('\n[3/7] Verifying Triton model loading...')
        await asyncio.sleep(3)  # Give Triton time to load

        # Model is loaded as {name}_trt_end2end for End2End format
        triton_model_name = f'{TRITON_NAME}_trt_end2end'
        response = await client.get(f'{TRITON_BASE}/v2/models/{triton_model_name}')
        if response.status_code != 200:
            print(f'❌ Model not found in Triton: {response.status_code}')
            return False

        model_info = response.json()
        print('✅ Model loaded in Triton!')
        print(f'  Name: {model_info["name"]}')
        print(f'  Versions: {model_info.get("versions", [])}')

        # Step 4: Check model ready state
        print('\n[4/7] Checking model ready state...')
        response = await client.get(f'{TRITON_BASE}/v2/models/{TRITON_NAME}/ready')
        if response.status_code != 200:
            print(f'❌ Model not ready: {response.status_code}')
            return False

        print('✅ Model is ready for inference!')

        # Step 5: Test through /detect endpoint (normal API pipeline)
        print('\n[5/7] Testing UPLOADED MODEL through /detect endpoint...')
        print(f'  Using model: {triton_model_name}')
        print('  This is the model we just uploaded and converted!')

        if not TEST_IMAGE.exists():
            print(f'❌ Test image not found: {TEST_IMAGE}')
            return False

        with open(TEST_IMAGE, 'rb') as f:
            files = {'image': (TEST_IMAGE.name, f, 'image/jpeg')}
            params = {'model_name': triton_model_name}  # ← USING UPLOADED MODEL

            response = await client.post(
                f'{API_BASE}/detect',
                files=files,
                params=params,
            )

        if response.status_code != 200:
            print(f'❌ Detection failed: {response.status_code}')
            print(f'  Response: {response.text}')
            return False

        detections = response.json()
        num_dets = len(detections.get('detections', []))
        inference_time = detections.get('inference_time_ms', 0)
        model_used = detections.get('model', {}).get('name', 'unknown')

        print('✅ Detection successful!')
        print(f'  Model used: {model_used}')
        if model_used != triton_model_name:
            print(f'  ❌ ERROR: Expected {triton_model_name} but got {model_used}')
            return False
        print('  ✅ Confirmed using uploaded model!')
        print(f'  Detections: {num_dets}')
        print(f'  Inference time: {inference_time:.2f}ms')

        if num_dets > 0:
            det = detections['detections'][0]
            print(f'  Top detection: {det["class_name"]} ({det["confidence"]:.2f})')

        # Step 5.5: List all models and confirm our uploaded model is there
        print('\n[5.5/8] Listing all models to confirm uploaded model exists...')
        response = await client.get(f'{API_BASE}/models')
        if response.status_code != 200:
            print(f'❌ Failed to list models: {response.status_code}')
            return False

        models_list = response.json()
        model_names = [m['name'] for m in models_list.get('models', [])]
        if triton_model_name in model_names:
            print('✅ Uploaded model found in models list!')
            model_info = next(m for m in models_list['models'] if m['name'] == triton_model_name)
            print(f'  Status: {model_info.get("status")}')
            print(f'  Backend: {model_info.get("backend")}')
            print(f'  Max batch: {model_info.get("max_batch_size")}')
            print(f'  Classes: {model_info.get("num_classes")}')
        else:
            print('❌ Uploaded model NOT found in models list!')
            print(f'  Available models: {model_names}')
            return False

        # Step 6: Test through /analyze endpoint (full pipeline)
        print('\n[6/8] Testing UPLOADED MODEL through /analyze endpoint...')
        print(f'  Using model: {triton_model_name}')

        with open(TEST_IMAGE, 'rb') as f:
            files = {'image': (TEST_IMAGE.name, f, 'image/jpeg')}
            params = {'model_name': triton_model_name}  # ← USING UPLOADED MODEL

            response = await client.post(
                f'{API_BASE}/analyze',
                files=files,
                params=params,
            )

        if response.status_code != 200:
            print(f'❌ Analysis failed: {response.status_code}')
            print(f'  Response: {response.text}')
            return False

        analysis = response.json()
        print('✅ Full analysis successful!')
        print(f'  Detections: {len(analysis.get("detections", []))}')
        print(f'  Faces: {len(analysis.get("faces", []))}')
        print(f'  Global embedding: {len(analysis.get("global_embedding", []))} dims')
        print(f'  OCR texts: {len(analysis.get("ocr_results", {}).get("texts", []))}')

        # Step 7: Clean up
        print('\n[7/8] Cleaning up test model...')
        response = await client.delete(f'{API_BASE}/models/{TRITON_NAME}')

        if response.status_code != 200:
            print(f'⚠️  Cleanup warning: {response.status_code}')
            print(f'  You may need to manually delete: models/{TRITON_NAME}/')
        else:
            print('✅ Test model cleaned up!')

        # Step 8: Show usage examples
        print('\n[8/8] Usage examples for the uploaded model...')
        print('\n' + '=' * 80)
        print('✅ ALL TESTS PASSED!')
        print('=' * 80)
        print('\n📋 UPLOADED MODEL USAGE:')
        print(f'  Model Name: {triton_model_name}')
        print(f'  Original File: {MODEL_PATH.name}')
        print('  Format: TensorRT End2End (GPU NMS)')
        print('\n🔧 HOW TO USE THIS MODEL:')
        print('\n  1. Object Detection:')
        print('     curl -X POST http://localhost:4603/detect \\')
        print('       -F "image=@your_image.jpg" \\')
        print(f'       -F "model_name={triton_model_name}"')
        print('\n  2. Full Analysis:')
        print('     curl -X POST http://localhost:4603/analyze \\')
        print('       -F "image=@your_image.jpg" \\')
        print(f'       -F "model_name={triton_model_name}"')
        print('\n  3. List All Models:')
        print('     curl http://localhost:4603/models')
        print('\n  4. Check Model Status:')
        print(
            f'     curl http://localhost:4603/models/{triton_model_name.replace("_trt_end2end", "")}/status'
        )
        print('\n  5. Delete Model:')
        print(
            f'     curl -X DELETE http://localhost:4603/models/{triton_model_name.replace("_trt_end2end", "")}'
        )

        print('\n📊 API WORKFLOW SUMMARY:')
        print('  1. ✅ Upload .pt files via POST /models/upload')
        print('  2. ✅ Export happens automatically in background')
        print('  3. ✅ Models auto-load into Triton when ready')
        print('  4. ✅ Models work through all API endpoints (/detect, /analyze)')
        print('  5. ✅ Model name appears in response to confirm usage')
        print('  6. ✅ Models listed via GET /models')
        print('  7. ✅ Models can be cleaned up via DELETE /models/{name}')

        return True


async def main():
    """Main entry point."""
    import sys

    try:
        success = await test_upload_workflow()
        sys.exit(0 if success else 1)
    except Exception as e:
        print('\n❌ Test failed with exception:')
        print(f'  {type(e).__name__}: {e}')
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
