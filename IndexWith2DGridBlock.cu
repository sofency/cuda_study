#include "common/common.h"
#include <cudia_runtime.h>
#include <stdio.h>

// 线程束分支会降低GPU实际的计算能力
// 线程束分支对程序性能影响通过分支效率衡量

// 设置内核函数
__global__ void sumMatrixOnGPU2D(int *arrayA, int* arrayB, int *arrayC, const int nx, const int ny)
{
  int ix  = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;

  unsigned int idx = iy * nx + ix;
  if (ix < nx && iy < ny) { // 当数据不是线程块整数倍时 会溢出 所以判断下下标是否正确
    arrayC[idx] = arrayA[idx] + arrayB[idx];
  }
}


int main(int argc, char **argv) {
  if (argc != 3) {
    return -1;
  }
  int block_x = atoi(argv[1]);
  int block_y = atoi(argv[2]);

  int nDeviceNumber = 0;
  cudaError_t error = ErrorCheck(cudaGetDeviceCount(&nDeviceNumber), __FILE__, __LINE__);
  if (error != cudaSuccess || nDeviceNumber == 0) {
    printf("no cuda\n");
    return 0;
  }

  int dev = 0;
  error = ErrorCheck(cudaSetDevice(dev), __FILE__, __LINE__);
  if (error != cudaSuccess) {
    printf("set error\n");
    return 0;
  }
  // 定义矩阵大小
  int nx = 1 << 14, ny = 1 << 14;
  int matrix_size = nx * ny;
  int bytes = matrix_size * sizeof(float);

  // 初始化矩阵数据 二维矩阵转换为一维矩阵
  int* array_host_A = (float*)malloc(bytes);
  int* array_host_B = (float*)malloc(bytes);
  int* array_host_res = (float*)malloc(bytes);

  // 初始化操作
  for (int i = 0; i < size; i++) {
    array_host_A[i] = i;
    array_host_B[i] = i + 1;
  }


  // 分配GPU内存
  int* array_device_A, *array_device_B, *array_host_C;
  cudaMalloc((void**)&array_device_A, bytes);
  cudaMalloc((void**)&array_device_B, bytes);
  cudaMalloc((void**)&array_device_C, bytes);


  // 主机数据拷贝到GPU
  cudaMemcpy(array_device_A, array_host_A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(array_device_B, array_host_B, bytes, cudaMemcpyHostToDevice);

  dim3 block(block_x, block_y);
  dim3 grid((nx-1) / block.x + 1, (ny-1) / block.y  + 1);
  printf("block(%d, %d), grid(%d, %d)\n", block.x, block.y, grid.x, grid.y);
  double begin = GetCPUSecond();
  sumMatrixOnGPU2D<<<grid, block>>>(array_device_A, array_device_B, array_device_C, nx, ny);
  cudaDeviceSynchronize();
  double end = GetCPUSecond();
  printf("element size %d, sum cost time %.5f", matrix_size, end - begin);

  cudaMemcpy(array_host_res, array_device_C, bytes, cudaMemcpyDeviceToHost);

  for (int i = 0; i< 10; i++) {
    printf("idx:%d, a:%d, b:%d, res:%d\n", i+1, array_host_A[i], array_host_B[i], array_host_res[i]);
  }

  cudaFree(array_device_A);
  cudaFree(array_device_B);
  cudaFree(array_device_C);

  free(array_host_A);
  free(array_host_B);
  free(array_host_res);
  cudaDeviceReset();
  return 0;
}