#include<cuda_runtime.h>
#include "common/common.h"
#include <stdio.h>

/**
 * @brief nvprof --help
 * sudo  /path/nvprof ./code 就可以看到程序运行时的信息
 * @param ip 
 * @param size 
 */
void initialData(float *ip, int size) 
{
  time_t t;
  srand((unsigned)time(&t));
  printf("Matrix is ");
  for (int i = 0; i< size; i++) {
    ip[i] = (float)(rand() & 0xFF) / 10.0f;
    printf("%.2f ", ip[i]);
  }
  printf("\n");
  return;
}

// 定义内核函数
__global__ void sumArraysOnGPU(float *A, float* B, float *C, const int N) {
  // threadIdx.x 表示当前线程块内的线程id blockIdx.x 表示当前是第几个线程块， blockDim.x表示32
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  // int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < N) {
    C[i] = A[i] + B[i];
  } 
}

int main(int argc, char **argv) {
  int nDeviceNumber = 0;
  // 检测当前设备与cuda兼容的设备
  cudaError_t error = ErrorCheck(cudaGetDeviceCount(&nDeviceNumber));
  if (error != cudaSuccess || nDeviceNumber == 0) {
    printf("NO CUDA camptable GPU found\n");
    return 0;
  }
  // set up device

  int dev = 0;
  // 设置GPU设备
  error = ErrorCheck(cudaSetDevice(dev));
  if (error != cudaSuccess) {
    printf("fail set GPU for computing\n");
    return 0;
  } else {
    printf("set GPU 0 for computing\n");
  }

  // 向左移动14位
  int nElem = 1 << 14;
  
  size_t bytes = nElem * sizeof(float);
  float *h_A, *h_B, *gpuRef;
  h_A = (float*)malloc(bytes);
  h_B = (float*)malloc(bytes);
  gpuRef = (float*)malloc(bytes);

  if (h_A && h_B && gpuRef) {
    printf("allocate memory successfully\n");
  } else {
    printf("allocate memory error\n");
    return 0;
  }

  initialData(h_A, nElem);
  initialData(h_B, nElem);
  memset(gpuRef, 0, bytes);
  // 分配GPU内存
  float *d_A, *d_B,*d_C;
  cudaMalloc((float**)&d_A, bytes);
  cudaMalloc((float**)&d_B, bytes);
  cudaMalloc((float**)&d_C, bytes);

  if (d_A==NULL || d_B==NULL || d_C==NULL) {
    printf("fail to allocate memory for GPU\n");
    free(h_A);
    free(h_B);
    free(gpuRef);
    return -1;
  }


  // 将主机数据拷贝到GPU
  cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, gpuRef, bytes, cudaMemcpyHostToDevice);


  dim3 block(32);
  dim3 grid(nElem/32);

  printf("Execution condigure <<%d,%d>>>. total element:%d\n", grid.x, block.x, nElem);

  double begin = GetCPUSecond();
  sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
  cudaDeviceSynchronize();
  double end = GetCPUSecond();

  cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost);

  printf("element size, matrix add cost %.5f\n", nElem, end - begin);

  for(int i = 0; i< 200, i++) {
    printf("maxtrix_A:%.2f, maxtrix_B:%.2f, result:%.2f\n", h_A[i], h_B[i], gpuRef[i]);
  }
  free(h_A);
  free(h_B);
  free(gpuRef);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaDeviceReset();

  return 0;





}