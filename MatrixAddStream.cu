#include<cuda_runtime.h>
#include "common/common.h"
#include <stdio.h>

/**
 * @brief nvprof --help
 * sudo  /path/nvprof ./code 就可以看到程序运行时的信息
 * 流事件的作用：同步流的执行 监控设备执行进度
 * CUDA流事件用于检测流的执行知否到达指定的操作点
 * 流事件插入流后，当其关联的操作完成后，流事件会在主机线程中产生一个完成标志
 * 主机线程同步 cudaEventSynchroize()
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
  ErrorCheck(cudaGetDeviceCount(&nDeviceNumber));
  // set up device
  int dev = 0;
  // 设置GPU设备
  ErrorCheck(cudaSetDevice(dev));

  // 检查是否支持全局内存 nvcc -Xptxas -dlcm=cg MatrixAdd.cu -o MatrixAdd 
  // -dlcm=cg 是另一个参数，它用来设置内存一致性模型。
  // 它可以用来设置PTX汇编级别的优化选项
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  if (deviceProp.globalL1CacheSupported) {
    printf("Global L1 Cache is supported\n");
  }

  // 向左移动24位
  int nElem = 1 << 24;
  
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

  // 获取设备优先级
  int lowPriority = 0;
  int highPriority = 0;
  cudaDeviceGetStreamPriorityRange(&lowPriority, &highPriority);
  printf("Priority Range is from %d to %d \n", lowPriority, highPriority);



  // 流化处理
  cudaStream_t data_stream;
  cudaStreamCreate(&data_stream);
  // 将主机数据通过流式拷贝到GPU 异步进行数据拷贝
  cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, data_stream);
  cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice, data_stream);
  
  // TODO 只是创建事件
  cudaEvent_t cp_evt;
  ErrorCheck(cudaEventCreate(&cp_evt));
  ErrorCheck(cudaEventRecord(cp_evt, data_stream));
  cudaEventSynchroize(cp_evt); // 等待数据拷贝完成

  // cudaStreamSynchronize(data_stream);
  

  dim3 block(512);
  dim3 grid((nElem - 1) / block.x + 1, 1);

  printf("Execution condigure <<%d,%d>>>. total element:%d\n", grid.x, block.x, nElem);

  double begin = GetCPUSecond();
  
  // 这样流失执行
  cudaStream_t kernel_stream;
  // 设置优先级
  cudaStreamCreateWithPriority(&kernel_stream, cudaStreamDefault, highPriority);
  sumArraysOnGPU<<<grid, block, 0, kernel_stream>>>(d_A, d_B, d_C, nElem);

  cudaEvent_t cp_end_evt;
  ErrorCheck(cudaEventCreate(&cp_end_evt));
  ErrorCheck(cudaEventRecord(cp_end_evt, kernel_stream));
  cudaEventSynchroize(cp_end_evt); // 等待数据拷贝完成
  printf("sum execute succcess\n");

  // cudaDeviceSynchronize(); 或者上面事件监督
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
  cudaStreamDestory(data_stream);
  cudaStreamDestory(kernel_stream);

  cudaEventDestory(cp_evt);
  cudaEventDestory(cp_end_evt);
  cudaDeviceReset();

  return 0;





}