// Rust engine only, not read by the Python engine: planar RGB [3, h, w] to interleaved [h, w, 3],
// the copy CuPy makes of the transposed view in frame_engine/branches/derivatives.py, so a
// rendition reaches nvJPEG as NVJPEG_INPUT_RGBI. One thread per pixel.
extern "C" __global__ void planes_to_rgb(
    const unsigned char* src, unsigned char* dst, long long pixels)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixels) return;
    dst[i * 3] = src[i];
    dst[i * 3 + 1] = src[i + pixels];
    dst[i * 3 + 2] = src[i + 2 * pixels];
}
