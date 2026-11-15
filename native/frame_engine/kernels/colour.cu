
extern "C" __global__ void icc_gather(unsigned char* px, long long n, const unsigned char* table)
{
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        unsigned char* p = px + i * 3;
        const unsigned char* t = table + ((size_t)p[0] * 65536 + (size_t)p[1] * 256 + p[2]) * 3;
        p[0] = t[0]; p[1] = t[1]; p[2] = t[2];
    }
}
