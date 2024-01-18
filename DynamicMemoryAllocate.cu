#include <cuda_runtime.h>
#include "common/common.h"
#include <stdio.h>

extern __shared__ int dynamicArray[];
__global__ void dynamic_shared_mem()
{
  dynamic_shared_mem[threadIdx.x] = threadIdx.x + 1;
  printf("access dynamic arrcy, dynamic_shared_mem[%d] = %d\n", threadIdx.x, dynamic_shared_mem[threadIdx.x]);
}


int main(int argc, char**argv) 
{
  int nDeviceNumber = 0;
  ErrorCheck(cudaGetDeviceCount(&nDeviceNumber));

  int dev = 0;
  ErrorCheck(cudaSetDevice(dev));

  dim3 block(16);
  dim3 grid(1);

  // 第三个参数就是动态共享内存的大小
  dynamic_shared_mem<<<grid, block, grid.x * block.x * sizeof(int)>>>();
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}