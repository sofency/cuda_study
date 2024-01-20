#include "common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

void initialData(float *ip, int size) 
{
  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i< size; i++) {
    ip[i] = (float)(rand() & 0xFF) / 10.0f;
  }
}

// useData 就是回调函数的参数 第三个
void data_callback(cudaStream_t stream, cudaError_t status, void* userData) {
  printf("data callback datasize: %d\n", *((int*)useData));
}


int main(int argc, char**argv) {
  int nDeviceNumber = 0;
  // 检测当前设备与cuda兼容的设备
  ErrorCheck(cudaGetDeviceCount(&nDeviceNumber));
  // set up device
  int dev = 0;
  // 设置GPU设备
  ErrorCheck(cudaSetDevice(dev));

  int nElem = 1 << 12;
  
  size_t nBytes = nElem * sizeof(float);
  float *h_A = (float*)malloc(sizeof(float));
  initialData(h_A, nElem);

  float* d_A;
  cudaMalloc((float**)&d_A, nBytes);

  cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
  // 0 默认流  
  // data_callback 回掉函数
  // void* userData
  // 未来使用 当前必须是0
  cudaStreamAddCallback(0, data_callback, &nBytes, 0);


  free(h_A);
  cudaFree(d_A);
  cudaDeviceReset();

  return 0;
}