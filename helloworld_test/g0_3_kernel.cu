// kernel.cu — 你要补全的地方（点火仪式）
// 任务：每个线程负责一个下标 i，执行 out[i] = a[i] + 1.0f
//
// 全局下标公式（背下来，以后会用到一百次）：
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//
// TODO 1: 算出全局下标 i
// TODO 2: 边界守卫：只有 i < n 的线程才读写（grid 不能整除 N 时会有多余线程）
// TODO 3: 写 out[i] = a[i] + 1.0f

__global__ void addOne(const float* a, float* out, int n) {
    // 在这里写你的代码
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        out[i] = a[i] + 1.0f;
    }
}