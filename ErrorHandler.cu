#include<common/common.h>
#include<stdio.h>

// 获取GPU数量 cudaGetDeviceCount
// 设备需要使用的GPU cudaSetDevice
// 获取GPU信息 cudaGetDeviceProperties

int main(int argc, char **argv) 
{
  float* gpuMemory = NULL;
  // 出错 则__FILE__, __LINE__ 就会有错误的信息值
  ErrorCheck(cudaMalloc(&gpuMemory, sizeof(float)));
  ErrorCheck(cudaFree(gpuMemory));
  ErrorCheck(cudaFree(gpuMemory));
  ErrorCheck(cudaDeviceReset());
  return 0;
}