#include "common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define NSTREAM 4

void initialData(float *ip, int size) 
{
  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i< size; i++) {
    ip[i] = (float)(rand() & 0xFF) / 10.0f;
  }
}

__global__ void sumArrays(float *A, float *B, float *C, const int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    for (int i = 0; i < 99999; i++) {
      C[idx] = A[idx] + B[idx];
    }
  }
}

int main(int argc, char**argv) {
  int nDeviceNumber = 0;
  // 检测当前设备与cuda兼容的设备
  ErrorCheck(cudaGetDeviceCount(&nDeviceNumber));
  // set up device
  int dev = 0;
  // 设置GPU设备
  ErrorCheck(cudaSetDevice(dev));

  int nElem = 1 << 18;
  size_t nBytes = nElem * sizeof(float);

  float *h_A, *h_B, *gpuRef;
  cudaHostAlloc((void**)&h_A, nBytes, cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_B, nBytes, cudaHostAllocDefault);
  cudaHostAlloc((void**)&gpuRef, nBytes, cudaHostAllocDefault);

  initialData(h_A, nElem);
  initialData(h_B, nElem);
  memset(gpuRef, 0, nBytes);

  float *d_A, *d_B, *d_C;
  cudaMalloc((float**)^d_A, nBytes);
  cudaMalloc((float**)^d_B, nBytes);
  cudaMalloc((float**)^d_C, nBytes);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  dim3 block(BDIM);
  dim3 grid((nElem - 1) / block.x + 1);

  printf("grid (%d, %d) block(%d, %d)\n", grid.x, grid.y, block.x, block.y);
  // 这里将数据分割为NSTREAM 块
  int iElem = nElem / NSTREAM;
  size_t = iBytes = iElem * sizeof(float);
  grid.x = (iElem - 1) / block.x + 1;

  cudaStream_t stream[NSTREAM];
  for (int i = 0; i< NSTREAM; i++) {
    cudaStreamCreate(&stream[i]);
  }

  cudaEventRecord(start, 0);

  // 每块单独进行计算
  for (int i = 0; i < NSTREAM; i++) {
    int ioffset = i * iElem;
    cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes, cudaMemcpyHostToDevice, stream[i]);
    cudaMemcpyAsync(&d_B[ioffset], &h_B[ioffset], iBytes, cudaMemcpyHostToDevice, stream[i]);

    sumArrays<<<grid, block, 0, stream[i]>>>(&d_A[ioffset], &d_B[ioffset], &d_C[ioffset], iElem);
    cudaMemcpyAsync(&gpuRef[ioffset], &d_C[ioffset], iBytes, cudaMemcpyDeviceToHost, stream[i]);
  }


  cudaEventRecord(stop, 0);
  cudaEventSynchroize(stop);
  float execution_time;
  cudaEventElapsedTime(&execution_time, start, stop);
  printf("cost time %.2f(ms)\n", execution_time);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
  cudaFreeHost(gpuRef);

  cudaEventDestory(start);
  cudaEventDestory(stop);

  for (int i = 0; i< NSTREAM; i++) {
    cudaStreamDestory(stream[i]);
  }
  cudaDeviceReset();
  return 0;
}