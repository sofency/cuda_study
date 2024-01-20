#include "common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <omp.h>

/**
 * @brief nvcc -Xcompiler -fopenmp 00-stream_dispatch.cu -o stream_dispatch & ./stream_dispatch 
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
 
 #define NSTREAM 4
 
 __device__ void kernel_func()
 {
   double sum = 0.0;
   long i = 999999;
   while(i>0) {
     for (long j = 0; j < 999999; j++) {
       sum = sum + tan(0.1) * tan(0.1);
     }
     i = i - 1;
   }
 }
 
 __global__ void kernel(int stream, int id) {
   if (0 == threadIdx.x) {
     printf("kernel %d is executed in stream_%d\n", id, stream);
   }
   kernel_func();
 }
 
 int main(int argc, char**argv) {
   int nDeviceNumber = 0;
   // 检测当前设备与cuda兼容的设备
   ErrorCheck(cudaGetDeviceCount(&nDeviceNumber));
   // set up device
   int dev = 0;
   // 设置GPU设备
   ErrorCheck(cudaSetDevice(dev));
 
   float elapsed_time;
 
   int n_streams = NSTREAM;
   cudaStream_t* streams = (cudaStream_t*) malloc(n_streams * sizeof(cudaStream_t));
 
   for (int i = 0; i< n_streams; i++) {
     ErrorCheck(cudaStreamCreate(streams[i]));
   }
 
   dim3 block(1);
   dim3 grid(1);
 
   cudaEvent_t start, stop;
   ErrorCheck(cudaEventCreate(&start));
   ErrorCheck(cudaEventCreate(&stop));
 
   // record start
   ErrorCheck(cudaEventRecord(start, 0)); // 0表示走默认流方式
   
   // 使用openmp多线程分配流
   omp_set_num_threads(NSTREAM);
   #pragma omp parallel{
     // 获取线程id
     int threadId = omp_get_thread_num();
     kernel<<<grid, block, 0, streams[threadId]>>>(threadId, 1);
     kernel<<<grid, block, 0, streams[threadId]>>>(threadId, 2);
     kernel<<<grid, block, 0, streams[threadId]>>>(threadId, 3);
     kernel<<<grid, block, 0, streams[threadId]>>>(threadId, 4);
   }
   
     
   
 
   ErrorCheck(cudaEventRecord(stop, 0)); // 0表示走默认流方式
   ErrorCheck(cudaEventElapsedTime(&elapsed_time, start, stop));
 
   for (int i = 0; i< n_streams; i++) {
     ErrorCheck(cudaStreamDestory(streams[i]));
   }
 
   free(streams);
   ErrorCheck(cudaEventDestory(start));
   ErrorCheck(cudaEventDestory(stop));
   cudaDeviceReset();
   return 0;
 }
 
 