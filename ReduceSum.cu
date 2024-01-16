#include <cuda_runtime.h>
#include <stdio.h>
#include "common/common.h"

/**
 * @brief 邻域计算
 * 
 * @param input 
 * @param output 
 * @param size 
 * @return __global__ 
 */
__global__ void reduceNeighbor(int* input, int*output, int size) {
  unsigned tid = threadIdx.x;
  unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
  int *array = input + blockIdx.x * blockDim.x;
  if (idx > size) return;
  for (int stride = 1; stride < blockDim.x; stride*=2) {
    if ((tid % (stride * 2)) == 0) {
      array[tid] += array[tid + stride];
    }

    // 等待线程块中所有线程执行完毕后才向下执行
    __syncthreads();
  }

  // 写回
  if (tid == 0) output[blockIdx.x] = array[0];
}

/**
 * @brief 间域计算 仅适合数组个数为偶数
 * 
 * @param input 
 * @param output 
 * @param size 
 * @return __global__ 
 */
__global__ void reduce(int* input, int*output, int size) {
  unsigned tid = threadIdx.x;
  unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
  int *array = input + blockIdx.x * blockDim.x;
  if (idx > size) return;
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid <stride) {
      array[tid] += array[tid + stride];
    }
    // 等待线程块中所有线程执行完毕后才向下执行
    __syncthreads();
  }

  // 写回
  if (tid == 0) output[blockIdx.x] = array[0];
}

// 使用常量内存
__constant__ int length;
__global__ void reduceWithConstant(int* input, int*output) {
  unsigned tid = threadIdx.x;
  unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
  int *array = input + blockIdx.x * blockDim.x;
  if (idx > length) return;
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid <stride) {
      array[tid] += array[tid + stride];
    }
    // 等待线程块中所有线程执行完毕后才向下执行
    __syncthreads();
  }

  // 写回
  if (tid == 0) output[blockIdx.x] = array[0];
}

/**
 * @brief 循环展开
 * 
 * @param input 
 * @param output 
 * @param size 
 * @return __global__ 
 */
 __global__ void reduceUnrolling(int* input, int*output, int size) {
  unsigned tid = threadIdx.x;

  unsigned idx = threadIdx.x + blockIdx.x * blockDim.x * 2;
  int *array = input + blockIdx.x * blockDim.x * 2;
  if (idx + blockDim.x < size) {
    // 将相邻的两个块加起来
    input[idx] += input[idx+blockDim.x];
  }
  __syncthreads();

  // 在一个块内进行间域计算
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid <stride) {
      array[tid] += array[tid + stride];
    }
    // 等待线程块中所有线程执行完毕后才向下执行
    __syncthreads();
  }

  // 写回
  if (tid == 0) output[blockIdx.x] = array[0];
}

int main(int argc, char** argv) {
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

  // 这样讲常量拷贝到GPU常量内存中
  ErrorCheck(cudaMemcpyToSymbol(length, &size, size(int), 0, cudaMemcpyHostToDevice));

  int blocksize = 512;
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
  reduceNeighbor<<<grid, block>>>(array_device_input, array_device_output, size);
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