#include<cuda_runtime.h>
#include "common/common.h"
#include <stdio.h>


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

__global__ void infiniteKernel()
{
  while(true){}
}

// 如果主线程中存在的核函数未执行完毕 
int main(int argc, char **argv) {
  int nDeviceNumber = 0;
  // 检测当前设备与cuda兼容的设备
  ErrorCheck(cudaGetDeviceCount(&nDeviceNumber));
  // set up device
  int dev = 0;
  // 设置GPU设备
  ErrorCheck(cudaSetDevice(dev));

  int nElem = 32;

  dim3 block(nElem);
  dim3 grid(1);
  cudaStream_t kernel_stream;
  cudaStreamCreate(&kernel_stream);
  infiniteKernel<<<grid, block, 0, kernel_stream>>>();

  cudaEvent_t event;
  // cudaEventDefault 对于CPU的消耗比较高 持续不断占用CPU时间片去查询事件的状态
  // 
  // ErrorCheck(cudaEventCreateWithFlags(&event, cudaEventDefault)); 
  ErrorCheck(cudaEventCreateWithFlags(&event, cudaEventBlockingSync)); 

  ErrorCheck(cudaEventRecord(event, kernel_stream));


  cudaEventSynchroize(event);
  printf("event finish\n");


  cudaStreamDestory(kernel_stream);
  cudaEventDestory(event);
  cudaDeviceReset();
  return 0;
}