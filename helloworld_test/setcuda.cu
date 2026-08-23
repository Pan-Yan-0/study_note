#include <stdio.h>

int main(){
    // Set the device to GPU 0
    int iDeviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&iDeviceCount);
    if (error != cudaSuccess || iDeviceCount == 0) {
        printf("No CUDA campatable GPU found!\n");
        exit(-1);
    }
    printf("The count of GPUs is %d\n",iDeviceCount);

    // 设置执行
    int iDev = 0;
    error = cudaSetDevice(iDev);
    if (error != cudaSuccess) {
        printf("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
        exit(-1);
    }
    printf("set GPU 0 for execution\n");
    
    return 0;
}