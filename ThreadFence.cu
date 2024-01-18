#include "common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void printInfo()
{
  __shared__ float share = 0.0;

  // 第一块
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    share = 5.0;
  }

  if (blockIdx.x == 1 && threadIdx.x == 0) {
    share = 6.0;
  }

  // 这样才能确保对于每一线程块的内存同步 不这样设置 只有每个线程块第一个线程有效
  __threadfence_block();

  printf("current block:%d, thread:%d, value:%d\n", blockIdx.x, threadIdx.x, share);
}


__device__ int g_shared = 0;
__global__ void thread_grid_fence()
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id == 0) {
    g_shared = 5.0;
  }

  __threadfence(); // 不加 全部都是0
  printf("current block:%d, thread:%d, value:%d, threadId:%d\n", blockIdx.x, threadIdx.x, g_shared, id);
}


int main(int argc, char**argv)
{
  int nDeviceNumber = 0;
  // 检查是否支持cuda
  ErrorCheck(cudaGetDeviceCount(&nDeviceNumber));
  
  int dev = 0;
  // 设置使用哪个GPU
  ErrorCheck(cudaSetDevice(dev));

  dim3 grid(2);
  dim3 block(32);
  printInfo<<<grid, block>>>();
  cudaDeviceSynchronize();

  return 0;
}