
__device__ __forceinline__ void tap(
    const unsigned char* src, int H, int W, float sx, float sy, float* acc)
{
    float fx = floorf(sx), fy = floorf(sy);
    float ax = sx - fx, ay = sy - fy;
    int x0 = min(max((int)fx, 0), W - 1), x1 = min(max((int)fx + 1, 0), W - 1);
    int y0 = min(max((int)fy, 0), H - 1), y1 = min(max((int)fy + 1, 0), H - 1);
    const unsigned char* p00 = src + ((size_t)y0 * W + x0) * 3;
    const unsigned char* p01 = src + ((size_t)y0 * W + x1) * 3;
    const unsigned char* p10 = src + ((size_t)y1 * W + x0) * 3;
    const unsigned char* p11 = src + ((size_t)y1 * W + x1) * 3;
    float w00 = (1.f - ax) * (1.f - ay), w01 = ax * (1.f - ay);
    float w10 = (1.f - ax) * ay, w11 = ax * ay;
    for (int c = 0; c < 3; ++c)
        acc[c] += w00 * p00[c] + w01 * p01[c] + w10 * p10[c] + w11 * p11[c];
}

extern "C" __global__ void warp_lines(
    const unsigned char* src, int H, int W,
    const float* maps, const int* widths, const int* taps, int n_lines, int batch_w, int height,
    unsigned char pad, unsigned char* dst)
{
    int line = blockIdx.y;
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (line >= n_lines || o >= batch_w * height) return;
    int u = o % batch_w, v = o / batch_w;
    unsigned char* d = dst + (size_t)line * 3 * height * batch_w;
    size_t at = (size_t)v * batch_w + u, plane = (size_t)height * batch_w;
    if (u >= widths[line]) {
        d[at] = pad; d[at + plane] = pad; d[at + 2 * plane] = pad;
        return;
    }
    const float* m = maps + line * 9;
    int nx = taps[line * 2], ny = taps[line * 2 + 1];
    float acc[3] = {0.f, 0.f, 0.f};
    for (int j = 0; j < ny; ++j) {
        float y = (float)v + ((float)j + 0.5f) / (float)ny - 0.5f;
        for (int i = 0; i < nx; ++i) {
            float x = (float)u + ((float)i + 0.5f) / (float)nx - 0.5f;
            float den = m[6] * x + m[7] * y + m[8];
            tap(src, H, W, (m[0] * x + m[1] * y + m[2]) / den, (m[3] * x + m[4] * y + m[5]) / den, acc);
        }
    }
    float norm = 1.f / (float)(nx * ny);
    // RGB frame to the BGR planes PP-OCR reads
    d[at] = (unsigned char)fminf(fmaxf(rintf(acc[2] * norm), 0.f), 255.f);
    d[at + plane] = (unsigned char)fminf(fmaxf(rintf(acc[1] * norm), 0.f), 255.f);
    d[at + 2 * plane] = (unsigned char)fminf(fmaxf(rintf(acc[0] * norm), 0.f), 255.f);
}
