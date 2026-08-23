# CUDA 学习笔记

本仓库记录我学习 CUDA 的过程代码。学习方式是借助 AI 构建的交互式闯关网站
[ikunlearncu](https://ikunlearncu.ok.kimi.link/level/w0-l0-3) 进行关卡式学习，
每关的代码练习都提交到这里。

## 目录结构

```
helloworld_test/     # W0 入门关卡
├── cuda_hello.cu    # 第一个 CUDA 程序：Hello World from GPU
├── cuda_ex.cu       # CUDA 基础练习（编译出 PTX 观察中间产物）
├── setcuda.cu       # CUDA 环境/设备相关练习
├── g0_3_main.cu     # 关卡 W0-L0-3 出题方 harness（请勿修改）
├── g0_3_kernel.cu   # 关卡 W0-L0-3 学员实现：addOne 向量加一
├── whoiam.cu        # 线程身份识别练习
└── vector_add.cu    # 向量加法练习
```

## 编译与运行

需要安装 CUDA Toolkit（nvcc）。编译时注意 `-arch` 要匹配自己 GPU 的算力：

```bash
# 查看 GPU 算力
nvidia-smi --query-gpu=compute_cap --format=csv

# 编译（以 sm_75 为例，替换成自己的算力）
nvcc -arch=sm_75 helloworld_test/cuda_hello.cu -o helloworld_test/cuda_hello

# 运行
./helloworld_test/cuda_hello
```

如果运行时报 `no kernel image`，就是 `-arch` 参数和 GPU 算力不匹配。

## 核心知识点速记

CUDA kernel 中最常用的全局下标公式：

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) { /* 边界守卫，grid 不能整除 N 时防越界 */ }
```

## 版本管理说明

- 编译产物（无扩展名的可执行文件）、PTX/cubin/o 中间文件不入库，见 `.gitignore`
- `.vscode/` 中包含 C/C++ IntelliSense 配置（CUDA 头文件路径、`.cu` 文件关联）
