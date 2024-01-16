#include <cuda_runtime.h>
#include <stdio.h>
#include "common/common.h"

/**
 * @brief 循环展开
 * 
 * @param input 
 * @param output 
 * @param size 
 * @return __global__ 
 */
 template <unsigned int IBlockSize>
 __global__ void reduceUnrolling(int* input, int*output, int size) {
  unsigned tid = threadIdx.x;

  unsigned idx = threadIdx.x + blockIdx.x * blockDim.x * 8;
  int *array = input + blockIdx.x * blockDim.x * 8;
  if (idx + 7 * blockDim.x < size) {
    // 将相邻的两个块加起来
    nt sum = 0;
    for (int i = 0; i < 8; ++i) {
      sum += input[idx + i * blockDim.x];
    }
    // 每8个合并一起处理
    input[idx] = sum;
  }
  __syncthreads();
  if (IBlockSize >= 1024 && tid < 512) array[tid] += array[tid + 512];
  __syncthreads();
  if (IBlockSize >= 512 && tid < 256) array[tid] += array[tid + 256];
  __syncthreads();
  if (IBlockSize >= 256 && tid < 128) array[tid] += array[tid + 128];
  __syncthreads();
  if (IBlockSize >= 128 && tid < 64) array[tid] += array[tid + 64];
  __syncthreads();
  
  if (tid < 32) {
    volatile int *vsmem = array;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
  }

  // 写回
  if (tid == 0) output[blockIdx.x] = array[0];
}

int main(int argc, char** argv) {
  // 根据传的参数设置块大小
  int blocksize = atoi(argv[1]);

  int nDeviceNumber = 0;
  cudaError_t error = ErrorCheck(cudaGetDeviceCount(&nDeviceNumber));
  if (error != cudaSuccess) {
    printf("no cuda for computing\n");
    return 0;
  }

  int dev = 0;
  error = ErrorCheck(cudaSetDevice(dev));
  if (error != cudaSuccess) {
    printf("no available cuda\n");
    return 0;
  }

  // 定义数组
  int size = 1 << 24;

  dim3 block(blocksize, 1);
  dim3 grid((size - 1) / block.x + 1, 1);
  printf("grid: %d, block: %d\n", grid.x, block.x);


  size_t bytes = size * sizeof(int);
  // 定义主机上的数组
  int* array_host = (int*)malloc(bytes);
  int* array_host_grid = (int*)malloc(grid.x * sizeof(int));

  for (int i = 0; i < size; i++) {
    array_host[i] = (int)(rand() & 0xFF);
  }
  
  int* array_device_input = NULL;
  int* array_device_output = NULL;
  cudaMalloc((void**)&array_device_input, bytes);
  cudaMalloc((void**)&array_device_output, grid.x * sizeof(int));

  // 数据由内存拷贝到GPU
  cudaMemcpy(array_device_input, array_host, bytes, cudaMemcpyHostToDevice);

  double begin = GetCPUSecond();
  switch(blocksize) {
    case 1024: reduceUnrolling<1024><<<grid.x/8, block>>>(array_device_input, array_device_output, size); break;
    case 512: reduceUnrolling<512><<<grid.x/8, block>>>(array_device_input, array_device_output, size); break;
    case 256: reduceUnrolling<256><<<grid.x/8, block>>>(array_device_input, array_device_output, size); break;
    case 64: reduceUnrolling<64><<<grid.x/8, block>>>(array_device_input, array_device_output, size); break;
  }
  cudaDeviceSynchronize();
  double iElaps = GetCPUSecond() - begin;

  cudaMemcpy(array_host_grid, array_device_output, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  int gpu_sum = 0;
  for (int i = 0; i< grid.x;i++) gpu_sum += array_host_grid[i];

  cudaFree(array_device_input);
  cudaFree(array_device_output);

  free(array_host);
  free(array_host_grid);
  return 0;
}