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

__global__ void init()
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

  int size = 1 << 24;
  int bytes = size * sizeof(float);
  float *h_data = (float*)malloc(bytes);
  initialData(h_data, size);

  init<<<2,size>>>(); // 下面设置了非阻塞流会异步执行 如果设置阻塞流 待等到该函数执行完毕才执行拷贝

  float *d_data;
  // 分配GPU内存
  cudaMalloc((float**)&d_data, bytes);

  cudaStream_t data_stream;
  cudaStreamCreate(&data_stream); // blocking stream
  // cudaStreamCreateWithFlags(&data_stream, cudaStreamNonBlocking); 非阻塞流

  cudaEvent_t begin_event;
  ErrorCheck(cudaEventCreate(&begin_event));
  ErrorCheck(cudaEventRecord(begin_event, data_stream));
  ErrorCheck(cudaMemcpyAsync(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice, data_stream));

  cudaEvent_t end_event;
  ErrorCheck(cudaEventCreate(&end_event));
  ErrorCheck(cudaEventRecord(end_event, data_stream));
  cudaEventSynchronize(end_event);
  printf("copy finish\n"); //非阻塞流 会执行

  float timeElapse = 0.0;
  cudaEventElapsedTime(&timeElapse, begin_event, end_event);
  printf("time copy from host to device cost %.2f(ms)\n", timeElapse);


  cudaDeviceSynchronize();
  free(h_data);
  cudaFree(d_data);
  cudaEventDestory(begin_event);
  cudaEventDestory(end_event);
  cudaDeviceReset();
  return 0;
}