#include <stdio.h>

__global__ void helloFromGPU(void){
    printf("Hello World from GPU!\n");
}

int main(void){
    // Launch kernel
    helloFromGPU<<<4, 4>>>();
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    printf("Hello World from CPU!\n");
    return 0;
}