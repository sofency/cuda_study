#include<cuda_runtime.h>
#include<stdio.h>
#include "common/common.h"

__managed__ float y = 9.0;

__global__ void unifiedMemory(float *A) 
{
  *A += y;
  printf("GPU unified_mem:%.2f\n", *A);
}


__global__ void pageLockedMemory(float *pin) {
  printf("GPU page-locked mem: %.2f\n", *pin);
}

__global__ void zeroCopyMemory(float *pin) {
  printf("GPU zero copy mem: %.2f\n", *pin);
}

int main(int argc, char** argv)
{
  int nDeviceNumber = 0;
  // 检查是否支持cuda
  ErrorCheck(cudaGetDeviceCount(&nDeviceNumber));
  
  int dev = 0;
  // 设置使用哪个GPU
  ErrorCheck(cudaSetDevice(dev));

  int supportManagedMemory = 0;
  // 检查是否支持统一内存
  ErrorCheck(cudaDeviceGetAttribute(&supportManagedMemory, cudaDevAttrManagedMemory, dev));
  if (supportManagedMemory == 0) {
    printf("allocate managed memory error\n");
    return 0;
  }
  dim3 block(1), grid(1);
  float *unified_mem = NULL;
  // 分配统一内存空间
  ErrorCheck(cudaMallocManaged((void**)&unified_mem));

  *unified_mem = 5.7;
  unifiedMemory<<<grid, block>>>(unified_mem);
  cudaDeviceSynchronize();
  printf("cpu unified memory: %.2f\n", *unified_mem);

  // 页锁定内存
  float *h_pinmem = NULL;
  // 主机上分配页锁定内存
  ErrorCheck(cudaMallocHost((float**)&h_pinmem, sizeof(float)));
  *h_pinmem = 4.8; //在主机上设置值
  pageLockedMemory<<<grid, block>>>(h_pinmem);
  cudaDeviceSynchronize();
  cudaFreeHost(h_pinmem);

  // 零拷贝内存
  float *zeroNum = NULL;
  ErrorCheck(cudaHostAlloc((float**)&zeroNum, sizeof(float)));
  *zeroNum = 13.2;
  zeroCopyMemory<<<grid, block>>>(zeroNum);
  cudaDeviceSynchronize();
  

  cudaFree();
  cudaDeviceReset();
  return 0;
}