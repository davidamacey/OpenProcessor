
extern "C" __global__ void resample_h(
    const unsigned char* src, int H, int W,
    int x0, int y0, int rows,
    const int* xmin, const int* xcnt, const float* wx, int kmax, int out_w,
    float* tmp)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    if (ox >= out_w || r >= rows) return;
    int sy = y0 + r;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f;
    if (sy >= 0 && sy < H) {
        const unsigned char* row = src + (size_t)sy * W * 3;
        int b = xmin[ox], n = xcnt[ox];
        const float* w = wx + (size_t)ox * kmax;
        for (int k = 0; k < n; ++k) {
            int sx = x0 + b + k;
            if (sx < 0 || sx >= W) continue;
            float wk = w[k];
            const unsigned char* p = row + sx * 3;
            a0 += wk * p[0]; a1 += wk * p[1]; a2 += wk * p[2];
        }
    }
    float* t = tmp + ((size_t)r * out_w + ox) * 3;
    t[0] = a0; t[1] = a1; t[2] = a2;
}

extern "C" __global__ void resample_v(
    const float* tmp, int out_w,
    const int* ymin, const int* ycnt, const float* wy, int kmax, int out_h, int ybase,
    unsigned char* dst, long long plane, int row_stride)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y;
    if (ox >= out_w || oy >= out_h) return;
    int b = ymin[oy] - ybase, n = ycnt[oy];
    const float* w = wy + (size_t)oy * kmax;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f;
    for (int k = 0; k < n; ++k) {
        const float* t = tmp + ((size_t)(b + k) * out_w + ox) * 3;
        float wk = w[k];
        a0 += wk * t[0]; a1 += wk * t[1]; a2 += wk * t[2];
    }
    size_t o = (size_t)oy * row_stride + ox;
    dst[o] = (unsigned char)fminf(fmaxf(rintf(a0), 0.f), 255.f);
    dst[o + plane] = (unsigned char)fminf(fmaxf(rintf(a1), 0.f), 255.f);
    dst[o + 2 * plane] = (unsigned char)fminf(fmaxf(rintf(a2), 0.f), 255.f);
}

__device__ __forceinline__ int refl(int i, int n) {
    if (n == 1) return 0;
    if (i < 0) return -i;
    if (i >= n) return 2 * n - 2 - i;
    return i;
}

__device__ __forceinline__ int grey(const unsigned char* src, int W, int x, int y) {
    const unsigned char* p = src + ((size_t)y * W + x) * 3;
    return (p[0] * 9798 + p[1] * 19235 + p[2] * 3735 + 16384) >> 15;
}

extern "C" __global__ void lap_sums(
    const unsigned char* src, int H, int W,
    const int* rects, int nrects, long long* out)
{
    int rid = blockIdx.y;
    if (rid >= nrects) return;
    int x1 = rects[rid * 4], y1 = rects[rid * 4 + 1], x2 = rects[rid * 4 + 2], y2 = rects[rid * 4 + 3];
    int w = x2 - x1, h = y2 - y1;
    long long n = (long long)w * h;
    long long s = 0, q = 0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        int x = (int)(i % w), y = (int)(i / w);
        int c = grey(src, W, x1 + x, y1 + y);
        int l = grey(src, W, x1 + refl(x - 1, w), y1 + y) + grey(src, W, x1 + refl(x + 1, w), y1 + y)
              + grey(src, W, x1 + x, y1 + refl(y - 1, h)) + grey(src, W, x1 + x, y1 + refl(y + 1, h))
              - 4 * c;
        s += l; q += (long long)l * l;
    }
    __shared__ long long ss[256];
    __shared__ long long sq[256];
    ss[threadIdx.x] = s; sq[threadIdx.x] = q;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) { ss[threadIdx.x] += ss[threadIdx.x + k]; sq[threadIdx.x] += sq[threadIdx.x + k]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd((unsigned long long*)&out[rid * 2], (unsigned long long)ss[0]);
        atomicAdd((unsigned long long*)&out[rid * 2 + 1], (unsigned long long)sq[0]);
    }
}
