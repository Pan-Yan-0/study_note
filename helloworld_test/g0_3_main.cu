// main.cu — 出题方 harness：初始化 + 发射 + 校验 + 打印凭证（请勿修改）
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

// 学员在 kernel.cu 里实现：
__global__ void addOne(const float* a, float* out, int n);

int main() {
    const int N = 256;  // 正好一个 block

    float *a, *out;
    // 统一内存：CPU/GPU 都能直接访问（内存管理的细节 W2 再讲，现在先用着）
    cudaMallocManaged(&a, N * sizeof(float));
    cudaMallocManaged(&out, N * sizeof(float));

    // 确定性输入：A[i] = i*0.5f（checksum 可手工推导，见 README.md）
    for (int i = 0; i < N; ++i) {
        a[i] = i * 0.5f;
        out[i] = 0.0f;
    }

    addOne<<<1, N>>>(a, out, N);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("QUEST_FAIL cudaError=%s\n", cudaGetErrorString(err));
        printf("提示：no kernel image = Makefile 的 ARCH 没改成你的算力\n");
        return 1;
    }

    double sum = 0.0;
    for (int i = 0; i < N; ++i) sum += fabs((double)out[i]);

    printf("out[0]=%.1f out[255]=%.1f（期望 1.0 和 128.5）\n", out[0], out[255]);
    printf("QUEST_PASS checksum=%.6f\n", sum);

    cudaFree(a);
    cudaFree(out);
    return 0;
}