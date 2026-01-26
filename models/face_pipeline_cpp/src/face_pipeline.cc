// Face Pipeline C++ Backend for Triton Inference Server
//
// High-performance face detection + recognition:
// 1. Receive face_images (640x640) and original_image (HD) via gRPC
// 2. Call YOLO11-face internally (BLS)
// 3. Transform boxes to HD coordinates (GPU)
// 4. Crop faces from HD image using GPU bilinear interpolation
// 5. Call ArcFace internally (BLS)
// 6. L2 normalize embeddings (GPU)
// 7. Return embeddings

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

// External CUDA kernel declarations
extern "C" {
void launchBilinearCrop(const float* src, float* dst, const float* boxes,
                        int src_h, int src_w, int num_faces, cudaStream_t stream);
void launchExpandBoxes(float* boxes, int num_faces, int img_h, int img_w,
                       cudaStream_t stream);
void launchTransformBoxes(const float* boxes_yolo, float* boxes_hd, int num_faces,
                          float scale, float pad_x, float pad_y,
                          int hd_h, int hd_w, cudaStream_t stream);
void launchL2Normalize(float* embeddings, int num_faces, int embed_dim,
                       cudaStream_t stream);
}

namespace triton { namespace backend { namespace face_pipeline {

// Constants
constexpr int YOLO_SIZE = 640;
constexpr int ARCFACE_SIZE = 112;
constexpr int EMBED_DIM = 512;
constexpr int MAX_FACES = 128;
constexpr float CONF_THRESHOLD = 0.5f;

//
// BLS Response Context - for synchronous inference
//
struct BLSResponseContext {
    std::mutex mtx;
    std::condition_variable cv;
    bool done{false};
    TRITONSERVER_InferenceResponse* response{nullptr};
    TRITONSERVER_Error* error{nullptr};
};

// Response callback for BLS - called when inference completes
static void BLSResponseCallback(
    TRITONSERVER_InferenceResponse* response,
    const uint32_t flags,
    void* userp)
{
    BLSResponseContext* ctx = static_cast<BLSResponseContext*>(userp);

    if (response != nullptr) {
        TRITONSERVER_Error* err = TRITONSERVER_InferenceResponseError(response);
        if (err != nullptr) {
            ctx->error = err;
            TRITONSERVER_InferenceResponseDelete(response);
            ctx->response = nullptr;
        } else {
            ctx->response = response;
            ctx->error = nullptr;
        }
    }

    {
        std::lock_guard<std::mutex> lk(ctx->mtx);
        ctx->done = true;
    }
    ctx->cv.notify_one();
}

// Release callback for BLS request - called when request can be freed
static void BLSReleaseCallback(
    TRITONSERVER_InferenceRequest* request,
    const uint32_t flags,
    void* userp)
{
    // Request will be deleted by caller after this callback
}

// Response allocator for BLS - allocates CPU memory for output tensors
static TRITONSERVER_Error*
BLSResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
    // Always allocate on CPU for BLS
    if (byte_size == 0) {
        *buffer = nullptr;
        *buffer_userp = nullptr;
    } else {
        *buffer = malloc(byte_size);
        if (*buffer == nullptr) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                "Failed to allocate memory for BLS response");
        }
        *buffer_userp = nullptr;
    }
    *actual_memory_type = TRITONSERVER_MEMORY_CPU;
    *actual_memory_type_id = 0;
    return nullptr;
}

// Release callback for response allocator
static TRITONSERVER_Error*
BLSResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
    if (buffer != nullptr) {
        free(buffer);
    }
    return nullptr;
}

//
// ModelState - shared configuration
//
class ModelState : public BackendModel {
public:
    static TRITONSERVER_Error* Create(TRITONBACKEND_Model* triton_model,
                                       ModelState** state);
    virtual ~ModelState();

    const std::string& YoloModelName() const { return yolo_model_name_; }
    const std::string& ArcfaceModelName() const { return arcface_model_name_; }
    TRITONSERVER_Server* Server() const { return server_; }
    TRITONSERVER_ResponseAllocator* BLSAllocator() const { return bls_allocator_; }

private:
    ModelState(TRITONBACKEND_Model* triton_model);
    TRITONSERVER_Error* Initialize();

    std::string yolo_model_name_{"yolo11_face_small_trt_end2end"};
    std::string arcface_model_name_{"arcface_w600k_r50"};
    TRITONSERVER_Server* server_{nullptr};
    TRITONSERVER_ResponseAllocator* bls_allocator_{nullptr};
};

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model) {}

ModelState::~ModelState() {
    if (bls_allocator_ != nullptr) {
        TRITONSERVER_ResponseAllocatorDelete(bls_allocator_);
    }
}

TRITONSERVER_Error*
ModelState::Initialize() {
    RETURN_IF_ERROR(TRITONBACKEND_ModelServer(TritonModel(), &server_));

    // Create BLS response allocator
    RETURN_IF_ERROR(TRITONSERVER_ResponseAllocatorNew(
        &bls_allocator_, BLSResponseAlloc, BLSResponseRelease, nullptr));

    return nullptr;
}

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state) {
    try {
        *state = new ModelState(triton_model);
        RETURN_IF_ERROR((*state)->Initialize());
    } catch (const BackendModelException& ex) {
        RETURN_ERROR_IF_TRUE(
            ex.err_ != nullptr, TRITONSERVER_ERROR_INTERNAL,
            std::string("failed to create model state: ") +
            TRITONSERVER_ErrorMessage(ex.err_));
    }
    return nullptr;
}

//
// ModelInstanceState - per-instance GPU resources
//
class ModelInstanceState : public BackendModelInstance {
public:
    static TRITONSERVER_Error* Create(
        ModelState* model_state,
        TRITONBACKEND_ModelInstance* triton_model_instance,
        ModelInstanceState** state);
    virtual ~ModelInstanceState();

    TRITONSERVER_Error* ProcessRequests(
        TRITONBACKEND_Request** requests, uint32_t request_count);

private:
    ModelInstanceState(ModelState* model_state,
                       TRITONBACKEND_ModelInstance* triton_model_instance);

    // BLS inference helpers
    TRITONSERVER_Error* CallYolo(
        const float* face_images,
        int32_t* num_dets, float* det_boxes, float* det_scores);

    TRITONSERVER_Error* CallArcface(
        const float* face_crops, int num_faces, float* embeddings);

    ModelState* model_state_;
    ::cudaStream_t stream_;
    int device_id_;

    // Pre-allocated GPU buffers
    float* d_boxes_yolo_;
    float* d_boxes_hd_;
    float* d_face_crops_;
    float* d_embeddings_;

    // Pre-allocated CPU buffers
    std::vector<float> h_boxes_yolo_;
    std::vector<float> h_boxes_hd_;
    std::vector<float> h_face_crops_;
    std::vector<float> h_embeddings_;
    std::vector<float> h_face_images_;  // For BLS input (always CPU)
};

ModelInstanceState::ModelInstanceState(
    ModelState* model_state,
    TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state),
      stream_(nullptr),
      device_id_(0),
      d_boxes_yolo_(nullptr),
      d_boxes_hd_(nullptr),
      d_face_crops_(nullptr),
      d_embeddings_(nullptr) {

    h_boxes_yolo_.resize(MAX_FACES * 4);
    h_boxes_hd_.resize(MAX_FACES * 4);
    h_face_crops_.resize(MAX_FACES * 3 * ARCFACE_SIZE * ARCFACE_SIZE);
    h_embeddings_.resize(MAX_FACES * EMBED_DIM);
    h_face_images_.resize(1 * 3 * YOLO_SIZE * YOLO_SIZE);  // For BLS input
}

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state,
    TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state) {

    *state = new ModelInstanceState(model_state, triton_model_instance);

    TRITONSERVER_InstanceGroupKind instance_kind;
    int32_t device_id;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(triton_model_instance, &instance_kind));
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(triton_model_instance, &device_id));

    if (instance_kind == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
        (*state)->device_id_ = device_id;
        cudaSetDevice(device_id);
    }

    cudaStreamCreate(&(*state)->stream_);

    cudaMalloc(&(*state)->d_boxes_yolo_, MAX_FACES * 4 * sizeof(float));
    cudaMalloc(&(*state)->d_boxes_hd_, MAX_FACES * 4 * sizeof(float));
    cudaMalloc(&(*state)->d_face_crops_,
               MAX_FACES * 3 * ARCFACE_SIZE * ARCFACE_SIZE * sizeof(float));
    cudaMalloc(&(*state)->d_embeddings_, MAX_FACES * EMBED_DIM * sizeof(float));

    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        (std::string("Face pipeline C++ instance initialized on GPU ") +
         std::to_string(device_id)).c_str());

    return nullptr;
}

ModelInstanceState::~ModelInstanceState() {
    if (stream_) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }
    if (d_boxes_yolo_) cudaFree(d_boxes_yolo_);
    if (d_boxes_hd_) cudaFree(d_boxes_hd_);
    if (d_face_crops_) cudaFree(d_face_crops_);
    if (d_embeddings_) cudaFree(d_embeddings_);
}

TRITONSERVER_Error*
ModelInstanceState::CallYolo(
    const float* face_images,
    int32_t* num_dets, float* det_boxes, float* det_scores)
{
    TRITONSERVER_Server* server = model_state_->Server();
    const std::string& model_name = model_state_->YoloModelName();

    // Create inference request
    TRITONSERVER_InferenceRequest* request = nullptr;
    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestNew(
        &request, server, model_name.c_str(), -1));

    // Set release callback
    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestSetReleaseCallback(
        request, BLSReleaseCallback, nullptr));

    // Set input
    const int64_t input_shape[] = {1, 3, YOLO_SIZE, YOLO_SIZE};
    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestAddInput(
        request, "images", TRITONSERVER_TYPE_FP32, input_shape, 4));

    size_t input_size = 1 * 3 * YOLO_SIZE * YOLO_SIZE * sizeof(float);
    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestAppendInputData(
        request, "images", face_images, input_size,
        TRITONSERVER_MEMORY_CPU, 0));

    // Request outputs
    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestAddRequestedOutput(request, "num_dets"));
    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestAddRequestedOutput(request, "det_boxes"));
    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestAddRequestedOutput(request, "det_scores"));

    // Create response context
    BLSResponseContext ctx;

    // Set response callback with allocator for BLS output tensors
    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestSetResponseCallback(
        request, model_state_->BLSAllocator(), nullptr, BLSResponseCallback, &ctx));

    // Execute async
    RETURN_IF_ERROR(TRITONSERVER_ServerInferAsync(server, request, nullptr));

    // Wait for response
    {
        std::unique_lock<std::mutex> lk(ctx.mtx);
        ctx.cv.wait(lk, [&ctx] { return ctx.done; });
    }

    // Check for errors
    if (ctx.error != nullptr) {
        TRITONSERVER_InferenceRequestDelete(request);
        return ctx.error;
    }

    // Extract outputs
    if (ctx.response != nullptr) {
        // Get num_dets
        const void* num_dets_data;
        size_t num_dets_size;
        TRITONSERVER_MemoryType mem_type;
        int64_t mem_id;
        const char* name;
        TRITONSERVER_DataType dtype;
        const int64_t* shape;
        uint64_t dims;
        void* userp;

        RETURN_IF_ERROR(TRITONSERVER_InferenceResponseOutput(
            ctx.response, 0, &name, &dtype, &shape, &dims,
            &num_dets_data, &num_dets_size, &mem_type, &mem_id, &userp));

        *num_dets = *static_cast<const int32_t*>(num_dets_data);

        // Get det_boxes if there are detections
        if (*num_dets > 0) {
            const void* boxes_data;
            size_t boxes_size;

            RETURN_IF_ERROR(TRITONSERVER_InferenceResponseOutput(
                ctx.response, 1, &name, &dtype, &shape, &dims,
                &boxes_data, &boxes_size, &mem_type, &mem_id, &userp));

            int copy_count = std::min(*num_dets, static_cast<int32_t>(MAX_FACES));
            memcpy(det_boxes, boxes_data, copy_count * 4 * sizeof(float));
        }

        TRITONSERVER_InferenceResponseDelete(ctx.response);
    }

    TRITONSERVER_InferenceRequestDelete(request);
    return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::CallArcface(
    const float* face_crops, int num_faces, float* embeddings)
{
    if (num_faces == 0) return nullptr;

    TRITONSERVER_Server* server = model_state_->Server();
    const std::string& model_name = model_state_->ArcfaceModelName();

    TRITONSERVER_InferenceRequest* request = nullptr;
    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestNew(
        &request, server, model_name.c_str(), -1));

    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestSetReleaseCallback(
        request, BLSReleaseCallback, nullptr));

    const int64_t input_shape[] = {num_faces, 3, ARCFACE_SIZE, ARCFACE_SIZE};
    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestAddInput(
        request, "input.1", TRITONSERVER_TYPE_FP32, input_shape, 4));

    size_t input_size = num_faces * 3 * ARCFACE_SIZE * ARCFACE_SIZE * sizeof(float);
    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestAppendInputData(
        request, "input.1", face_crops, input_size,
        TRITONSERVER_MEMORY_CPU, 0));

    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestAddRequestedOutput(request, "683"));

    BLSResponseContext ctx;
    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestSetResponseCallback(
        request, model_state_->BLSAllocator(), nullptr, BLSResponseCallback, &ctx));

    RETURN_IF_ERROR(TRITONSERVER_ServerInferAsync(server, request, nullptr));

    {
        std::unique_lock<std::mutex> lk(ctx.mtx);
        ctx.cv.wait(lk, [&ctx] { return ctx.done; });
    }

    if (ctx.error != nullptr) {
        TRITONSERVER_InferenceRequestDelete(request);
        return ctx.error;
    }

    if (ctx.response != nullptr) {
        const void* emb_data;
        size_t emb_size;
        TRITONSERVER_MemoryType mem_type;
        int64_t mem_id;
        const char* name;
        TRITONSERVER_DataType dtype;
        const int64_t* shape;
        uint64_t dims;
        void* userp;

        RETURN_IF_ERROR(TRITONSERVER_InferenceResponseOutput(
            ctx.response, 0, &name, &dtype, &shape, &dims,
            &emb_data, &emb_size, &mem_type, &mem_id, &userp));

        memcpy(embeddings, emb_data, num_faces * EMBED_DIM * sizeof(float));
        TRITONSERVER_InferenceResponseDelete(ctx.response);
    }

    TRITONSERVER_InferenceRequestDelete(request);
    return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, uint32_t request_count) {

    for (uint32_t r = 0; r < request_count; r++) {
        TRITONBACKEND_Request* request = requests[r];
        TRITONBACKEND_Response* response = nullptr;
        RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));

        try {
            // Get input tensors
            TRITONBACKEND_Input* face_images_input;
            TRITONBACKEND_Input* original_image_input;
            TRITONBACKEND_Input* orig_shape_input;

            RETURN_IF_ERROR(TRITONBACKEND_RequestInput(
                request, "face_images", &face_images_input));
            RETURN_IF_ERROR(TRITONBACKEND_RequestInput(
                request, "original_image", &original_image_input));
            RETURN_IF_ERROR(TRITONBACKEND_RequestInput(
                request, "orig_shape", &orig_shape_input));

            // Get properties
            const int64_t* orig_image_shape;
            uint32_t orig_image_dims;
            uint64_t orig_image_byte_size;
            RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
                original_image_input, nullptr, nullptr, &orig_image_shape,
                &orig_image_dims, &orig_image_byte_size, nullptr));

            // Get buffers
            const void* face_images_buffer;
            uint64_t face_images_size;
            TRITONSERVER_MemoryType face_mem_type;
            int64_t face_mem_id;
            RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
                face_images_input, 0, &face_images_buffer, &face_images_size,
                &face_mem_type, &face_mem_id));

            const void* orig_image_buffer;
            uint64_t orig_image_size;
            TRITONSERVER_MemoryType orig_mem_type;
            int64_t orig_mem_id;
            RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
                original_image_input, 0, &orig_image_buffer, &orig_image_size,
                &orig_mem_type, &orig_mem_id));

            const void* orig_shape_buffer;
            uint64_t orig_shape_size;
            TRITONSERVER_MemoryType shape_mem_type;
            int64_t shape_mem_id;
            RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
                orig_shape_input, 0, &orig_shape_buffer, &orig_shape_size,
                &shape_mem_type, &shape_mem_id));

            // Parse dims
            int hd_h = static_cast<int>(orig_image_shape[2]);
            int hd_w = static_cast<int>(orig_image_shape[3]);
            const int32_t* orig_shape = static_cast<const int32_t*>(orig_shape_buffer);
            int orig_h = orig_shape[0];
            int orig_w = orig_shape[1];

            // Letterbox transform params
            float scale = std::min(static_cast<float>(YOLO_SIZE) / orig_h,
                                  static_cast<float>(YOLO_SIZE) / orig_w);
            float pad_x = (YOLO_SIZE - orig_w * scale) / 2;
            float pad_y = (YOLO_SIZE - orig_h * scale) / 2;

            // STEP 1: Call YOLO
            // BLS requires CPU memory, so copy to host buffer if needed
            const size_t face_images_byte_size = 1 * 3 * YOLO_SIZE * YOLO_SIZE * sizeof(float);
            if (face_mem_type == TRITONSERVER_MEMORY_GPU) {
                cudaMemcpy(h_face_images_.data(), face_images_buffer,
                          face_images_byte_size, cudaMemcpyDeviceToHost);
            } else {
                memcpy(h_face_images_.data(), face_images_buffer, face_images_byte_size);
            }

            int32_t num_faces = 0;
            TRITONSERVER_Error* yolo_err = CallYolo(
                h_face_images_.data(), &num_faces, h_boxes_yolo_.data(), nullptr);
            if (yolo_err != nullptr) {
                LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                    (std::string("YOLO BLS failed: ") +
                     TRITONSERVER_ErrorMessage(yolo_err)).c_str());
                TRITONSERVER_ErrorDelete(yolo_err);
                num_faces = 0;
            }

            num_faces = std::min(num_faces, static_cast<int32_t>(MAX_FACES));

            if (num_faces > 0) {
                // STEP 2: Transform boxes to HD coords
                cudaMemcpyAsync(d_boxes_yolo_, h_boxes_yolo_.data(),
                               num_faces * 4 * sizeof(float),
                               cudaMemcpyHostToDevice, stream_);

                launchTransformBoxes(
                    d_boxes_yolo_, d_boxes_hd_, num_faces,
                    scale, pad_x, pad_y, hd_h, hd_w, stream_);

                // STEP 3: Expand boxes with margin
                launchExpandBoxes(d_boxes_hd_, num_faces, hd_h, hd_w, stream_);

                // STEP 4: GPU crop faces
                const float* hd_image = static_cast<const float*>(orig_image_buffer);
                float* d_hd_image = nullptr;

                if (orig_mem_type == TRITONSERVER_MEMORY_GPU) {
                    d_hd_image = const_cast<float*>(hd_image);
                } else {
                    cudaMalloc(&d_hd_image, orig_image_byte_size);
                    cudaMemcpyAsync(d_hd_image, hd_image, orig_image_byte_size,
                                   cudaMemcpyHostToDevice, stream_);
                }

                launchBilinearCrop(
                    d_hd_image, d_face_crops_, d_boxes_hd_,
                    hd_h, hd_w, num_faces, stream_);

                // STEP 5: Call ArcFace
                cudaMemcpyAsync(h_face_crops_.data(), d_face_crops_,
                               num_faces * 3 * ARCFACE_SIZE * ARCFACE_SIZE * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_);
                cudaStreamSynchronize(stream_);

                TRITONSERVER_Error* arc_err = CallArcface(
                    h_face_crops_.data(), num_faces, h_embeddings_.data());
                if (arc_err != nullptr) {
                    LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                        (std::string("ArcFace BLS failed: ") +
                         TRITONSERVER_ErrorMessage(arc_err)).c_str());
                    TRITONSERVER_ErrorDelete(arc_err);
                }

                // STEP 6: L2 normalize
                cudaMemcpyAsync(d_embeddings_, h_embeddings_.data(),
                               num_faces * EMBED_DIM * sizeof(float),
                               cudaMemcpyHostToDevice, stream_);

                launchL2Normalize(d_embeddings_, num_faces, EMBED_DIM, stream_);

                // Copy results back
                cudaMemcpyAsync(h_embeddings_.data(), d_embeddings_,
                               num_faces * EMBED_DIM * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_);
                cudaMemcpyAsync(h_boxes_hd_.data(), d_boxes_hd_,
                               num_faces * 4 * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_);

                cudaStreamSynchronize(stream_);

                if (orig_mem_type != TRITONSERVER_MEMORY_GPU && d_hd_image) {
                    cudaFree(d_hd_image);
                }
            }

            // Create outputs
            std::vector<int64_t> num_faces_shape = {1};
            TRITONBACKEND_Output* num_faces_output;
            RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
                response, &num_faces_output, "num_faces",
                TRITONSERVER_TYPE_INT32, num_faces_shape.data(), 1));

            void* num_faces_buf;
            uint64_t num_faces_buf_sz;
            TRITONSERVER_MemoryType num_faces_mt;
            int64_t num_faces_mi;
            RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(
                num_faces_output, &num_faces_buf, sizeof(int32_t),
                &num_faces_mt, &num_faces_mi));
            *static_cast<int32_t*>(num_faces_buf) = num_faces;

            // Boxes
            std::vector<int64_t> boxes_shape = {num_faces, 4};
            TRITONBACKEND_Output* boxes_output;
            RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
                response, &boxes_output, "boxes",
                TRITONSERVER_TYPE_FP32, boxes_shape.data(), 2));
            if (num_faces > 0) {
                void* boxes_buf;
                uint64_t boxes_buf_sz;
                TRITONSERVER_MemoryType boxes_mt;
                int64_t boxes_mi;
                RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(
                    boxes_output, &boxes_buf, num_faces * 4 * sizeof(float),
                    &boxes_mt, &boxes_mi));
                memcpy(boxes_buf, h_boxes_hd_.data(), num_faces * 4 * sizeof(float));
            }

            // Landmarks (placeholder)
            std::vector<int64_t> lm_shape = {num_faces, 10};
            TRITONBACKEND_Output* lm_output;
            RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
                response, &lm_output, "landmarks",
                TRITONSERVER_TYPE_FP32, lm_shape.data(), 2));
            if (num_faces > 0) {
                void* lm_buf;
                uint64_t lm_buf_sz;
                TRITONSERVER_MemoryType lm_mt;
                int64_t lm_mi;
                RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(
                    lm_output, &lm_buf, num_faces * 10 * sizeof(float),
                    &lm_mt, &lm_mi));
                memset(lm_buf, 0, num_faces * 10 * sizeof(float));
            }

            // Scores (placeholder)
            std::vector<int64_t> sc_shape = {num_faces};
            TRITONBACKEND_Output* sc_output;
            RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
                response, &sc_output, "scores",
                TRITONSERVER_TYPE_FP32, sc_shape.data(), 1));
            if (num_faces > 0) {
                void* sc_buf;
                uint64_t sc_buf_sz;
                TRITONSERVER_MemoryType sc_mt;
                int64_t sc_mi;
                RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(
                    sc_output, &sc_buf, num_faces * sizeof(float),
                    &sc_mt, &sc_mi));
                std::fill_n(static_cast<float*>(sc_buf), num_faces, 1.0f);
            }

            // Embeddings
            std::vector<int64_t> emb_shape = {num_faces, EMBED_DIM};
            TRITONBACKEND_Output* emb_output;
            RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
                response, &emb_output, "embeddings",
                TRITONSERVER_TYPE_FP32, emb_shape.data(), 2));
            if (num_faces > 0) {
                void* emb_buf;
                uint64_t emb_buf_sz;
                TRITONSERVER_MemoryType emb_mt;
                int64_t emb_mi;
                RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(
                    emb_output, &emb_buf, num_faces * EMBED_DIM * sizeof(float),
                    &emb_mt, &emb_mi));
                memcpy(emb_buf, h_embeddings_.data(), num_faces * EMBED_DIM * sizeof(float));
            }

            // Quality (placeholder)
            std::vector<int64_t> q_shape = {num_faces};
            TRITONBACKEND_Output* q_output;
            RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
                response, &q_output, "quality",
                TRITONSERVER_TYPE_FP32, q_shape.data(), 1));
            if (num_faces > 0) {
                void* q_buf;
                uint64_t q_buf_sz;
                TRITONSERVER_MemoryType q_mt;
                int64_t q_mi;
                RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(
                    q_output, &q_buf, num_faces * sizeof(float),
                    &q_mt, &q_mi));
                std::fill_n(static_cast<float*>(q_buf), num_faces, 1.0f);
            }

            RETURN_IF_ERROR(TRITONBACKEND_ResponseSend(
                response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr));

        } catch (const std::exception& e) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                (std::string("Error: ") + e.what()).c_str());
            TRITONBACKEND_ResponseDelete(response);
            RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
            RETURN_IF_ERROR(TRITONBACKEND_ResponseSend(
                response, TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, e.what())));
        }
    }

    return nullptr;
}

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend) {
    const char* name;
    RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &name));
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        (std::string("TRITONBACKEND_Initialize: ") + name).c_str());
    return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend) {
    return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model) {
    const char* name;
    RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &name));
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        (std::string("TRITONBACKEND_ModelInitialize: ") + name).c_str());

    ModelState* model_state;
    RETURN_IF_ERROR(ModelState::Create(model, &model_state));
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(
        model, reinterpret_cast<void*>(model_state)));

    return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model) {
    void* state;
    RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &state));
    delete reinterpret_cast<ModelState*>(state);
    return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance) {
    const char* name;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &name));
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name).c_str());

    TRITONBACKEND_Model* model;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

    void* model_state_ptr;
    RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &model_state_ptr));
    ModelState* model_state = reinterpret_cast<ModelState*>(model_state_ptr);

    ModelInstanceState* instance_state;
    RETURN_IF_ERROR(ModelInstanceState::Create(model_state, instance, &instance_state));
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
        instance, reinterpret_cast<void*>(instance_state)));

    return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance) {
    void* state;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &state));
    delete reinterpret_cast<ModelInstanceState*>(state);
    return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance,
    TRITONBACKEND_Request** requests,
    const uint32_t request_count) {

    void* state;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &state));
    ModelInstanceState* instance_state =
        reinterpret_cast<ModelInstanceState*>(state);

    return instance_state->ProcessRequests(requests, request_count);
}

}  // extern "C"

}}}  // namespace triton::backend::face_pipeline
