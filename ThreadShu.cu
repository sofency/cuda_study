/**
 * @file ThreadShu.cu
 * @author your name (you@domain.com)
 * @brief 线程束的优化 编译器对线程分支的优化能力有限，只有当分支下的代码量很少时优化才会起作用
 * @version 0.1
 * @date 2024-01-15
 * nvcc -g -G ThreadShu.cu -o ThreadShu     // -g -G 表示不使用cuda 默认的分支优化器
 * sudo /path/to/nvprof --metrics branch_efficiency ./ThreadShu
 * 如果代码中的条件判断值与线程id关联，则以线程数为基本单元访问数据
 *
 * 线程束计算资源包括 程序计数器、寄存器、共享内存
 * 线程数所需的计算资源属于片上资源，因此GPU在不同线程数间切换的成本可以忽略不计
 * 流处理器具有多个32位的寄存器和一定量的共享内存

 * 分配原则 
 * 如果SM中的资源无法满足至少一个线程块的需求,内核无法运行
 * 
 * 
 * @copyright Copyright (c) 2024
 * 
 */
 #include <cuda_runtime.h>
 #include <stdio.h>
 #include "common/common.h"

 __global__ void mathKernel(float *array) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  bool ipred = (tid % 2 == 0);
  if (ipred) {
    ia = 100.0f;
  }
  if (!pred) {
    ib = 200.0f;
  }
  array[tid] = ia + ib;
 }

 __global__ void mathKernel2(float *array) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  // 每一个线程束为一个执行单元
  if ((tid / wrapSize) % 2 == 0) {
    ia = 100.0f;
  } else {
    ib = 200.0f;
  }
  array[tid] = ia + ib;
 }

 int main(int argc, char **argv) {
  int nDeviceNumber = 0;
  cudaError_t error = ErrorCheck(cudaGetDeviceCount(&nDeviceNumber));
  if (error != cudaSuccess || nDeviceNumber == 0) {
    printf("no cuda\n");
    return 0;
  }

  int dev = 0;
  error = ErrorCheck(cudaSetDevice(dev));
  if (error != cudaSuccess) {
    printf("set error\n");
    return 0;
  }

  // 定义矩阵大小
  int size = 64;
  int blocksize = 64;
  dim3 block(blocksize, 1);
  dim3 grid((size - 1) / block.x + 1, 1);

  // 分配GPU内存
  float* array_device_A;
  size_t bytes = size * sizeof(float);
  error = ErrorCheck(cudaMalloc((void**)&array_device_A, bytes));
  if (error != cudaSuccess) {
    printf("allocate memory error\n");
    return 0;
  }

  mathKernel<<<grid, block>>>(array_device_A);
  cudaDeviceSynchronize();

  cudaFree(array_device_A);
  cudaDeviceReset();
  return 0;
}