// GPU Face Cropping Kernel
//
// Bilinear interpolation for high-quality face cropping on GPU.
// Replaces CPU cv2.resize() with GPU kernel.
//
// Input: HD image [3, H, W] on GPU
// Output: Face crops [N, 3, 112, 112] on GPU
// Boxes: [N, 4] face boxes in HD coordinates

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

// Constants
constexpr int ARCFACE_SIZE = 112;
constexpr float FACE_MARGIN = 0.4f;

// Bilinear interpolation kernel for face cropping
// Each thread handles one pixel in output
__global__ void bilinearCropKernel(
    const float* __restrict__ src,       // [3, src_h, src_w] source HD image
    float* __restrict__ dst,              // [num_faces, 3, 112, 112] output crops
    const float* __restrict__ boxes,      // [num_faces, 4] boxes [x1,y1,x2,y2] with margin applied
    const int src_h,
    const int src_w,
    const int num_faces)
{
    // Grid: (num_faces, ceil(112*112/256))
    // Block: (256)
    const int face_idx = blockIdx.x;
    const int pixel_idx = blockIdx.y * blockDim.x + threadIdx.x;
    const int crop_size = ARCFACE_SIZE;

    if (face_idx >= num_faces || pixel_idx >= crop_size * crop_size)
        return;

    const int out_y = pixel_idx / crop_size;
    const int out_x = pixel_idx % crop_size;

    // Get box coordinates (already expanded with margin)
    const float x1 = boxes[face_idx * 4 + 0];
    const float y1 = boxes[face_idx * 4 + 1];
    const float x2 = boxes[face_idx * 4 + 2];
    const float y2 = boxes[face_idx * 4 + 3];

    // Map output pixel to source coordinates using bilinear interpolation
    // +0.5 for pixel center alignment
    const float src_x = x1 + (x2 - x1) * (out_x + 0.5f) / crop_size;
    const float src_y = y1 + (y2 - y1) * (out_y + 0.5f) / crop_size;

    // Compute interpolation coordinates
    const int x0 = static_cast<int>(floorf(src_x));
    const int y0 = static_cast<int>(floorf(src_y));
    const int x1i = min(x0 + 1, src_w - 1);
    const int y1i = min(y0 + 1, src_h - 1);
    const int x0c = max(0, min(x0, src_w - 1));
    const int y0c = max(0, min(y0, src_h - 1));

    const float fx = src_x - x0;
    const float fy = src_y - y0;

    // Bilinear interpolation for each channel
    #pragma unroll
    for (int c = 0; c < 3; c++) {
        const float v00 = src[c * src_h * src_w + y0c * src_w + x0c];
        const float v01 = src[c * src_h * src_w + y0c * src_w + x1i];
        const float v10 = src[c * src_h * src_w + y1i * src_w + x0c];
        const float v11 = src[c * src_h * src_w + y1i * src_w + x1i];

        float value = v00 * (1.0f - fx) * (1.0f - fy) +
                      v01 * fx * (1.0f - fy) +
                      v10 * (1.0f - fx) * fy +
                      v11 * fx * fy;

        // ArcFace normalization: (pixel - 127.5) / 128.0
        // Input is [0, 1] normalized, so: (x * 255 - 127.5) / 128 = x * 1.9921875 - 0.99609375
        value = value * 1.9921875f - 0.99609375f;

        dst[face_idx * 3 * crop_size * crop_size +
            c * crop_size * crop_size +
            out_y * crop_size + out_x] = value;
    }
}

// Kernel to expand boxes with MTCNN-style margin and make square
__global__ void expandBoxesKernel(
    float* boxes,          // [num_faces, 4] modified in-place
    const int num_faces,
    const int img_h,
    const int img_w)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_faces) return;

    float x1 = boxes[idx * 4 + 0];
    float y1 = boxes[idx * 4 + 1];
    float x2 = boxes[idx * 4 + 2];
    float y2 = boxes[idx * 4 + 3];

    const float face_w = x2 - x1;
    const float face_h = y2 - y1;

    // MTCNN-style margin expansion (40% on each side)
    const float margin_w = face_w * FACE_MARGIN;
    const float margin_h = face_h * FACE_MARGIN;

    x1 -= margin_w;
    y1 -= margin_h;
    x2 += margin_w;
    y2 += margin_h;

    // Make square (expand shorter dimension)
    const float box_w = x2 - x1;
    const float box_h = y2 - y1;
    const float max_dim = fmaxf(box_w, box_h);
    const float cx = (x1 + x2) * 0.5f;
    const float cy = (y1 + y2) * 0.5f;

    x1 = cx - max_dim * 0.5f;
    y1 = cy - max_dim * 0.5f;
    x2 = cx + max_dim * 0.5f;
    y2 = cy + max_dim * 0.5f;

    // Clamp to image bounds
    boxes[idx * 4 + 0] = fmaxf(0.0f, fminf((float)img_w, x1));
    boxes[idx * 4 + 1] = fmaxf(0.0f, fminf((float)img_h, y1));
    boxes[idx * 4 + 2] = fmaxf(0.0f, fminf((float)img_w, x2));
    boxes[idx * 4 + 3] = fmaxf(0.0f, fminf((float)img_h, y2));
}

// Kernel to transform boxes from YOLO 640 space to HD coordinates
__global__ void transformBoxesKernel(
    const float* boxes_yolo,   // [num_faces, 4] normalized [0,1] in YOLO space
    float* boxes_hd,           // [num_faces, 4] pixel coords in HD space
    const int num_faces,
    const float scale,         // letterbox scale
    const float pad_x,         // letterbox x padding
    const float pad_y,         // letterbox y padding
    const int hd_h,
    const int hd_w)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_faces) return;

    const int YOLO_SIZE = 640;

    // Convert from normalized [0,1] to YOLO pixel coords
    float x1 = boxes_yolo[idx * 4 + 0] * YOLO_SIZE;
    float y1 = boxes_yolo[idx * 4 + 1] * YOLO_SIZE;
    float x2 = boxes_yolo[idx * 4 + 2] * YOLO_SIZE;
    float y2 = boxes_yolo[idx * 4 + 3] * YOLO_SIZE;

    // Inverse letterbox transformation
    x1 = (x1 - pad_x) / scale;
    y1 = (y1 - pad_y) / scale;
    x2 = (x2 - pad_x) / scale;
    y2 = (y2 - pad_y) / scale;

    // Clamp to HD image bounds
    boxes_hd[idx * 4 + 0] = fmaxf(0.0f, fminf((float)hd_w, x1));
    boxes_hd[idx * 4 + 1] = fmaxf(0.0f, fminf((float)hd_h, y1));
    boxes_hd[idx * 4 + 2] = fmaxf(0.0f, fminf((float)hd_w, x2));
    boxes_hd[idx * 4 + 3] = fmaxf(0.0f, fminf((float)hd_h, y2));
}

// L2 normalization kernel for embeddings
__global__ void l2NormalizeKernel(
    float* embeddings,     // [num_faces, embed_dim]
    const int num_faces,
    const int embed_dim)
{
    const int face_idx = blockIdx.x;
    if (face_idx >= num_faces) return;

    float* emb = embeddings + face_idx * embed_dim;

    // Compute L2 norm using parallel reduction
    __shared__ float shared_sum[256];
    float local_sum = 0.0f;

    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        local_sum += emb[i] * emb[i];
    }

    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Normalize
    const float norm = sqrtf(shared_sum[0] + 1e-10f);
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        emb[i] /= norm;
    }
}

// Host wrapper functions
extern "C" {

void launchBilinearCrop(
    const float* src,
    float* dst,
    const float* boxes,
    int src_h, int src_w,
    int num_faces,
    cudaStream_t stream)
{
    if (num_faces == 0) return;

    const int pixels_per_face = ARCFACE_SIZE * ARCFACE_SIZE;
    const int threads = 256;
    const int blocks_y = (pixels_per_face + threads - 1) / threads;

    dim3 grid(num_faces, blocks_y);
    dim3 block(threads);

    bilinearCropKernel<<<grid, block, 0, stream>>>(
        src, dst, boxes, src_h, src_w, num_faces);
}

void launchExpandBoxes(
    float* boxes,
    int num_faces,
    int img_h, int img_w,
    cudaStream_t stream)
{
    if (num_faces == 0) return;

    const int threads = 256;
    const int blocks = (num_faces + threads - 1) / threads;

    expandBoxesKernel<<<blocks, threads, 0, stream>>>(
        boxes, num_faces, img_h, img_w);
}

void launchTransformBoxes(
    const float* boxes_yolo,
    float* boxes_hd,
    int num_faces,
    float scale, float pad_x, float pad_y,
    int hd_h, int hd_w,
    cudaStream_t stream)
{
    if (num_faces == 0) return;

    const int threads = 256;
    const int blocks = (num_faces + threads - 1) / threads;

    transformBoxesKernel<<<blocks, threads, 0, stream>>>(
        boxes_yolo, boxes_hd, num_faces,
        scale, pad_x, pad_y, hd_h, hd_w);
}

void launchL2Normalize(
    float* embeddings,
    int num_faces,
    int embed_dim,
    cudaStream_t stream)
{
    if (num_faces == 0) return;

    l2NormalizeKernel<<<num_faces, 256, 0, stream>>>(
        embeddings, num_faces, embed_dim);
}

} // extern "C"
