// Face Pipeline C++ Backend for Triton Inference Server
//
// High-performance face detection + recognition with HD cropping:
// 1. YOLO11-face detection on 640x640 (fast)
// 2. Map boxes back to original HD coordinates
// 3. Crop faces from ORIGINAL HD image (preserves quality)
// 4. Resize to 112x112 for ArcFace
// 5. Extract 512-dim embeddings
//
// This is the industry standard approach used by MTCNN, InsightFace, RetinaFace.
// Cropping from HD preserves facial details for accurate recognition.

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>
#include <cstring>

namespace triton { namespace backend { namespace face_pipeline {

//
// Constants
//
constexpr int YOLO_SIZE = 640;
constexpr int ARCFACE_SIZE = 112;
constexpr int EMBED_DIM = 512;
constexpr int MAX_FACES = 128;
constexpr float FACE_MARGIN = 0.4f;  // MTCNN-style 40% margin
constexpr float CONF_THRESHOLD = 0.5f;

//
// CUDA Kernels for GPU-accelerated face cropping
//

// Bilinear interpolation for face cropping (runs on GPU)
__global__ void bilinearCropKernel(
    const float* __restrict__ src,      // [3, H, W] source image
    float* __restrict__ dst,             // [N, 3, 112, 112] output crops
    const float* __restrict__ boxes,     // [N, 4] boxes in src coordinates [x1,y1,x2,y2]
    int src_h, int src_w,
    int num_faces,
    int crop_size)                       // 112
{
    int face_idx = blockIdx.x;
    int pixel_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (face_idx >= num_faces || pixel_idx >= crop_size * crop_size)
        return;

    int out_y = pixel_idx / crop_size;
    int out_x = pixel_idx % crop_size;

    // Get box coordinates
    float x1 = boxes[face_idx * 4 + 0];
    float y1 = boxes[face_idx * 4 + 1];
    float x2 = boxes[face_idx * 4 + 2];
    float y2 = boxes[face_idx * 4 + 3];

    // Map output pixel to source coordinates (bilinear)
    float src_x = x1 + (x2 - x1) * (out_x + 0.5f) / crop_size;
    float src_y = y1 + (y2 - y1) * (out_y + 0.5f) / crop_size;

    // Bilinear interpolation
    int x0 = static_cast<int>(floorf(src_x));
    int y0 = static_cast<int>(floorf(src_y));
    int x1i = min(x0 + 1, src_w - 1);
    int y1i = min(y0 + 1, src_h - 1);
    x0 = max(0, min(x0, src_w - 1));
    y0 = max(0, min(y0, src_h - 1));

    float fx = src_x - x0;
    float fy = src_y - y0;

    // For each channel (RGB)
    for (int c = 0; c < 3; c++) {
        float v00 = src[c * src_h * src_w + y0 * src_w + x0];
        float v01 = src[c * src_h * src_w + y0 * src_w + x1i];
        float v10 = src[c * src_h * src_w + y1i * src_w + x0];
        float v11 = src[c * src_h * src_w + y1i * src_w + x1i];

        float value = v00 * (1 - fx) * (1 - fy) +
                      v01 * fx * (1 - fy) +
                      v10 * (1 - fx) * fy +
                      v11 * fx * fy;

        // ArcFace normalization: (x - 127.5) / 128.0
        // Input is [0, 1], so: (x * 255 - 127.5) / 128 = x * 1.9921875 - 0.99609375
        value = value * 1.9921875f - 0.99609375f;

        dst[face_idx * 3 * crop_size * crop_size + c * crop_size * crop_size + out_y * crop_size + out_x] = value;
    }
}

// L2 normalization kernel
__global__ void l2NormalizeKernel(
    float* embeddings,  // [N, 512]
    int num_faces,
    int embed_dim)
{
    int face_idx = blockIdx.x;
    if (face_idx >= num_faces) return;

    float* emb = embeddings + face_idx * embed_dim;

    // Compute L2 norm
    float norm_sq = 0.0f;
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        norm_sq += emb[i] * emb[i];
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        norm_sq += __shfl_down_sync(0xffffffff, norm_sq, offset);
    }

    // Thread 0 broadcasts norm
    __shared__ float shared_norm;
    if (threadIdx.x == 0) {
        shared_norm = sqrtf(norm_sq + 1e-10f);
    }
    __syncthreads();

    // Normalize
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        emb[i] /= shared_norm;
    }
}

//
// Model State - shared across instances
//
class ModelState : public BackendModel {
public:
    static TRITONSERVER_Error* Create(
        TRITONBACKEND_Model* triton_model, ModelState** state);
    virtual ~ModelState() = default;

    // Model names for BLS calls
    const char* yolo_model_name = "yolo11_face_small_trt_end2end";
    const char* arcface_model_name = "arcface_w600k_r50";

private:
    ModelState(TRITONBACKEND_Model* triton_model)
        : BackendModel(triton_model) {}
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state) {
    try {
        *state = new ModelState(triton_model);
    } catch (const BackendModelException& ex) {
        RETURN_ERROR_IF_TRUE(
            ex.err_ != nullptr, TRITONSERVER_ERROR_INTERNAL,
            std::string("failed to create model state: ") +
            TRITONSERVER_ErrorMessage(ex.err_));
    }
    return nullptr;
}

//
// Model Instance State - one per GPU instance
//
class ModelInstanceState : public BackendModelInstance {
public:
    static TRITONSERVER_Error* Create(
        ModelState* model_state,
        TRITONBACKEND_ModelInstance* triton_model_instance,
        ModelInstanceState** state);
    virtual ~ModelInstanceState();

    TRITONSERVER_Error* ProcessRequests(
        TRITONBACKEND_Request** requests, const uint32_t request_count);

private:
    ModelInstanceState(
        ModelState* model_state,
        TRITONBACKEND_ModelInstance* triton_model_instance)
        : BackendModelInstance(model_state, triton_model_instance),
          model_state_(model_state) {}

    // Internal BLS call to YOLO
    TRITONSERVER_Error* CallYOLO(
        const float* face_images,  // [B, 3, 640, 640]
        int batch_size,
        std::vector<int>& num_dets,
        std::vector<std::vector<float>>& boxes,
        std::vector<std::vector<float>>& scores);

    // Internal BLS call to ArcFace
    TRITONSERVER_Error* CallArcFace(
        const float* face_crops,  // [N, 3, 112, 112]
        int num_faces,
        float* embeddings);       // [N, 512]

    // Map boxes from YOLO 640 space to original HD coordinates
    void MapBoxesToOriginal(
        const std::vector<float>& boxes_640,  // [N, 4] normalized [0,1]
        const float* affine_matrix,            // [2, 3]
        int orig_h, int orig_w,
        std::vector<float>& boxes_orig);       // [N, 4] pixel coords

    // Expand boxes with MTCNN-style margin and make square
    void ExpandAndSquareBoxes(
        std::vector<float>& boxes,  // [N, 4] modified in-place
        int img_h, int img_w);

    ModelState* model_state_;
    cudaStream_t stream_;

    // GPU buffers
    float* d_face_crops_;    // [MAX_FACES, 3, 112, 112]
    float* d_embeddings_;    // [MAX_FACES, 512]
    float* d_boxes_;         // [MAX_FACES, 4]
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state,
    TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state) {
    try {
        *state = new ModelInstanceState(model_state, triton_model_instance);
    } catch (const BackendModelException& ex) {
        RETURN_ERROR_IF_TRUE(
            ex.err_ != nullptr, TRITONSERVER_ERROR_INTERNAL,
            std::string("failed to create instance state: ") +
            TRITONSERVER_ErrorMessage(ex.err_));
    }

    // Create CUDA stream
    cudaStreamCreate(&(*state)->stream_);

    // Allocate GPU buffers
    cudaMalloc(&(*state)->d_face_crops_,
        MAX_FACES * 3 * ARCFACE_SIZE * ARCFACE_SIZE * sizeof(float));
    cudaMalloc(&(*state)->d_embeddings_,
        MAX_FACES * EMBED_DIM * sizeof(float));
    cudaMalloc(&(*state)->d_boxes_,
        MAX_FACES * 4 * sizeof(float));

    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        "Face Pipeline C++ backend instance initialized with GPU buffers");

    return nullptr;
}

ModelInstanceState::~ModelInstanceState() {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    cudaFree(d_face_crops_);
    cudaFree(d_embeddings_);
    cudaFree(d_boxes_);
}

void ModelInstanceState::MapBoxesToOriginal(
    const std::vector<float>& boxes_640,
    const float* affine_matrix,
    int orig_h, int orig_w,
    std::vector<float>& boxes_orig)
{
    // affine_matrix format: [[scale, 0, pad_x], [0, scale, pad_y]]
    float scale = affine_matrix[0];  // [0,0]
    float pad_x = affine_matrix[2];  // [0,2]
    float pad_y = affine_matrix[5];  // [1,2]

    boxes_orig.resize(boxes_640.size());

    for (size_t i = 0; i < boxes_640.size(); i += 4) {
        // Boxes are normalized [0,1] in YOLO space
        float x1 = boxes_640[i + 0] * YOLO_SIZE;
        float y1 = boxes_640[i + 1] * YOLO_SIZE;
        float x2 = boxes_640[i + 2] * YOLO_SIZE;
        float y2 = boxes_640[i + 3] * YOLO_SIZE;

        // Inverse letterbox: orig_coord = (yolo_coord - pad) / scale
        boxes_orig[i + 0] = std::max(0.0f, std::min((float)orig_w, (x1 - pad_x) / scale));
        boxes_orig[i + 1] = std::max(0.0f, std::min((float)orig_h, (y1 - pad_y) / scale));
        boxes_orig[i + 2] = std::max(0.0f, std::min((float)orig_w, (x2 - pad_x) / scale));
        boxes_orig[i + 3] = std::max(0.0f, std::min((float)orig_h, (y2 - pad_y) / scale));
    }
}

void ModelInstanceState::ExpandAndSquareBoxes(
    std::vector<float>& boxes,
    int img_h, int img_w)
{
    for (size_t i = 0; i < boxes.size(); i += 4) {
        float x1 = boxes[i + 0];
        float y1 = boxes[i + 1];
        float x2 = boxes[i + 2];
        float y2 = boxes[i + 3];

        float face_w = x2 - x1;
        float face_h = y2 - y1;

        // MTCNN-style margin expansion (40% on each side)
        float margin_w = face_w * FACE_MARGIN;
        float margin_h = face_h * FACE_MARGIN;

        x1 -= margin_w;
        y1 -= margin_h;
        x2 += margin_w;
        y2 += margin_h;

        // Make square (expand shorter dimension)
        float box_w = x2 - x1;
        float box_h = y2 - y1;
        float max_dim = std::max(box_w, box_h);
        float cx = (x1 + x2) / 2;
        float cy = (y1 + y2) / 2;

        x1 = cx - max_dim / 2;
        y1 = cy - max_dim / 2;
        x2 = cx + max_dim / 2;
        y2 = cy + max_dim / 2;

        // Clamp to image bounds
        boxes[i + 0] = std::max(0.0f, std::min((float)img_w, x1));
        boxes[i + 1] = std::max(0.0f, std::min((float)img_h, y1));
        boxes[i + 2] = std::max(0.0f, std::min((float)img_w, x2));
        boxes[i + 3] = std::max(0.0f, std::min((float)img_h, y2));
    }
}

TRITONSERVER_Error*
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
        (std::string("Processing ") + std::to_string(request_count) + " requests").c_str());

    // Process each request
    for (uint32_t r = 0; r < request_count; r++) {
        TRITONBACKEND_Request* request = requests[r];

        // Create response
        TRITONBACKEND_Response* response = nullptr;
        RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));

        // Get inputs
        TRITONBACKEND_Input* face_images_input;
        TRITONBACKEND_Input* original_image_input;
        TRITONBACKEND_Input* orig_shape_input;
        TRITONBACKEND_Input* affine_matrix_input;

        RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, "face_images", &face_images_input));
        RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, "original_image", &original_image_input));
        RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, "orig_shape", &orig_shape_input));

        // affine_matrix is optional
        auto status = TRITONBACKEND_RequestInput(request, "affine_matrix", &affine_matrix_input);
        bool has_affine = (status == nullptr);

        // Get input data pointers
        const void* face_images_data;
        uint64_t face_images_size;
        TRITONSERVER_MemoryType face_images_mem_type;
        int64_t face_images_mem_id;
        RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
            face_images_input, 0, &face_images_data, &face_images_size,
            &face_images_mem_type, &face_images_mem_id));

        const void* original_image_data;
        uint64_t original_image_size;
        TRITONSERVER_MemoryType original_image_mem_type;
        int64_t original_image_mem_id;
        RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
            original_image_input, 0, &original_image_data, &original_image_size,
            &original_image_mem_type, &original_image_mem_id));

        const void* orig_shape_data;
        uint64_t orig_shape_size;
        TRITONSERVER_MemoryType orig_shape_mem_type;
        int64_t orig_shape_mem_id;
        RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
            orig_shape_input, 0, &orig_shape_data, &orig_shape_size,
            &orig_shape_mem_type, &orig_shape_mem_id));

        const int32_t* orig_shape = static_cast<const int32_t*>(orig_shape_data);
        int orig_h = orig_shape[0];
        int orig_w = orig_shape[1];

        // Get affine matrix if provided
        float affine_matrix[6] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
        if (has_affine) {
            const void* affine_data;
            uint64_t affine_size;
            TRITONSERVER_MemoryType affine_mem_type;
            int64_t affine_mem_id;
            RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
                affine_matrix_input, 0, &affine_data, &affine_size,
                &affine_mem_type, &affine_mem_id));
            memcpy(affine_matrix, affine_data, 6 * sizeof(float));
        }

        // TODO: Call YOLO via internal BLS
        // TODO: Map boxes to original coordinates
        // TODO: Crop faces from HD original using GPU kernel
        // TODO: Call ArcFace via internal BLS
        // TODO: L2 normalize embeddings

        // For now, return empty results (placeholder)
        int32_t num_faces = 0;
        std::vector<float> face_boxes(MAX_FACES * 4, 0.0f);
        std::vector<float> face_landmarks(MAX_FACES * 10, 0.0f);
        std::vector<float> face_scores(MAX_FACES, 0.0f);
        std::vector<float> face_embeddings(MAX_FACES * EMBED_DIM, 0.0f);
        std::vector<float> face_quality(MAX_FACES, 0.0f);

        // Create output tensors
        // ... (output creation code would go here)

        // Send response
        LOG_IF_ERROR(
            TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
            "failed to send response");
    }

    return nullptr;
}

//
// Triton Backend Lifecycle Functions
//

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend) {
    const char* cname;
    RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        (std::string("TRITONBACKEND_Initialize: ") + cname).c_str());

    // Check Triton version compatibility
    uint32_t api_version_major, api_version_minor;
    RETURN_IF_ERROR(TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        (std::string("Triton Backend API version: ") +
         std::to_string(api_version_major) + "." +
         std::to_string(api_version_minor)).c_str());

    return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend) {
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "TRITONBACKEND_Finalize: face_pipeline");
    return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model) {
    const char* cname;
    RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        (std::string("TRITONBACKEND_ModelInitialize: ") + cname).c_str());

    ModelState* model_state;
    RETURN_IF_ERROR(ModelState::Create(model, &model_state));
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(
        model, reinterpret_cast<void*>(model_state)));

    return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model) {
    void* vstate;
    RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
    ModelState* model_state = reinterpret_cast<ModelState*>(vstate);
    delete model_state;
    return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance) {
    const char* cname;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + cname).c_str());

    TRITONBACKEND_Model* model;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

    void* vmodelstate;
    RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
    ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

    ModelInstanceState* instance_state;
    RETURN_IF_ERROR(ModelInstanceState::Create(model_state, instance, &instance_state));
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
        instance, reinterpret_cast<void*>(instance_state)));

    return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance) {
    void* vstate;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
    ModelInstanceState* instance_state = reinterpret_cast<ModelInstanceState*>(vstate);
    delete instance_state;
    return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance,
    TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
    void* vstate;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
    ModelInstanceState* instance_state = reinterpret_cast<ModelInstanceState*>(vstate);

    return instance_state->ProcessRequests(requests, request_count);
}

}  // extern "C"

}}}  // namespace triton::backend::face_pipeline
