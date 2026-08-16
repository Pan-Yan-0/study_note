#include <cstdio>
#include <iostream>
__global__ void whoiam(void){
    std::cout << "我是 block" << blockIdx.x << " thread " << threadIdx.x << std::endl;
}

int main(void){
    return 0;
}