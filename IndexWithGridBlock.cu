#include "common/common.h"
#include <cudia_runtime.h>
#include <stdio.h>


// 初始化数据
void initialData(int* array, int size) {
  for (int i = 0; i < size; i++) {
    array[i] = i;
  }
}

__global__ void printThreadIndex(int *array, const int nx, const int ny)
{
  int ix  = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;

  unsigned int idx = iy * nx + ix;
  printf("thread_id(%d,%d) block_id(%d,%d) coordinate(%d,%d)"
          "global index %2d ival %2d\n",threadIdx.x,threadIdx.y,
          blockIdx.x,blockIdx.y, ix,iy, idx, array[idx]);
}
int main(int argc, char **argv) {
  // 定义矩阵大小
  int nx = 8, ny = 6;
  int matrix_size = nx * ny;
  int bytes = matrix_size * sizeof(float);

  // 初始化矩阵数据 二维矩阵转换为一维矩阵
  int* array_host = (float*)malloc(bytes);
  initialData(array_host, matrix_size);

  // 
  int* array_device = NULL;
  cudaMalloc((void**)&array_device, bytes);

  // 主机数据拷贝到GPU
  cudaMemcpy(array_device, array_host, bytes, cudaMemcpyHostToDevice);

  dim3 block(4,2);
  dim3 grid((nx-1) / block.x + 1, (ny-1) / block.y  + 1);
  printThreadIndex<<<grid, block>>>(array_device, nx, ny);
  cudaDeviceSynchronize();
  cudaFree(array_device);
  free(array_host);

  cudaDeviceReset();
  return 0;
}