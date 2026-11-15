
extern "C" __global__ void orient(
    const unsigned char* src, int H, int W, int orientation,
    unsigned char* dst, int out_h, int out_w)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    if (x >= out_w || y >= out_h) return;
    int xx = x, yy = y;
    if (orientation == 3 || orientation == 4 || orientation == 7 || orientation == 8) yy = out_h - 1 - yy;
    if (orientation == 2 || orientation == 3 || orientation == 6 || orientation == 7) xx = out_w - 1 - xx;
    int sx = xx, sy = yy;
    if (orientation >= 5 && orientation <= 8) { sx = yy; sy = xx; }
    const unsigned char* p = src + ((size_t)sy * W + sx) * 3;
    unsigned char* q = dst + ((size_t)y * out_w + x) * 3;
    q[0] = p[0]; q[1] = p[1]; q[2] = p[2];
}
