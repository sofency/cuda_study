#include "common/common.h"
#include <stdlib.h>

/**
 * @brief 默认开启了联合运算的优化 
 * nvcc --fmad=false file.cu -o file  // 将fmad优化关闭
 * @param out 
 * @param x 
 * @param value 
 * @return __global__ 
 */


__global__ void cas_kernel(int* out, int x, int value) {
  int old = atomicCAS(out, x, value);
  printf("original data is, %d\n", old);
}

__global__ void atomics(int *shared_var, int*values_read, int N, int iters)
{
  int i;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;
  values_read[tid] = atomicAdd(shared_var, 1);
  for (int i = 0; i < iters; i++) {
    atomicAdd(shared_var, 1);
  }
}

__global__ void unsafe(int *shared_var, int*values_read, int N, int iters) {
  int i;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;
  int old = *shared_var;
  *shared_var = old + 1;
  values_read[tid] = old;
  for (i = 0; i< itersl i++) {
    int old = *shared_var;
    *shared_var = old + 1;
  }
}

static void print_read_results(int* h_arr, int *d_arr, int M. const char*label)
{
  int i;
  cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
  printf("Threads performing %s operations read values", label);
  for (i = 0; i < N; i++) {
    printf(" %d", h_arr[i]);
  }
  printf("\n");
}

void testCAS()
{
  int* d_out, h_out;
  cudaMalloc((void**)&d_out, sizeof(int));
  cudaMemset(d_out, 0, sizeof(int));
  cas_kernel<<<1,1>>>(d_out, 1,5);
  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  printf("out is replaced, value=%d\n", h_out);
  cudaFree(d_out);
  cudaDeviceReset();
}

__global__ void self_defined_atomicAdd(int* address, int incr) {
  int guess = *address;
  int oldValue = atomicCAS(address, guess, guess + incr);
  while(oldValue != guess) {
    guess = oldValue;
    oldValue = atomicCAS(address, guess, guess + incr);
  }
}

void testPerformLoss() {
  int N = 64;
  int block = 32;
  int runs = 30;
  int iters = 100000;
  int r;
  int *d_shared_var;
  int h_shared_var_atomic, h_shared_var_unsafe;
  int *d_values_read_atomic;
  int *d_values_read_unsafe;
  int *h_values_read;

  cudaMalloc((void**)&d_shared_var, sizeof(int));
  cudaMalloc((void**)&d_values_read_atomic, N * sizeof(int));
  cudaMalloc((void**)&d_values_read_unsafe, N * sizeof(int));

  h_values_read = (int*)malloc(N * sizeof(int));

  double atomic_mean_time = 0;
  double unsafe_mean_time = 0;

  for (r = 0;r < runs; r++) {
    double start_atomic = GetCPUSecond();
    cudaMemset(d_shared_var, 0x00, sizeof(int));

    atomics<<<N / block, block>>>(d_shared_var, d_values_read_atomic, N, iters);
    cudaDeviceSynchronize();

    atomic_mean_time += GetCPUSecond() - start_atomic;
    cudaMemcpy(&h_shared_var_atomic, d_shared_var, sizeof(int), cudaMemcpyDeviceToHost);

    double start_unsafe = GetCPUSecond();
    cudaMemset(d_shared_var, 0x00, sizeof(int));

    atomics<<<N / block, block>>>(d_shared_var, d_values_read_unsafe, N, iters);
    cudaDeviceSynchronize();

    atomic_mean_time += GetCPUSecond() - start_unsafe;
    cudaMemcpy(&h_shared_var_unsafe, d_shared_var, sizeof(int), cudaMemcpyDeviceToHo
  }

  printf("In Total, %d runs using atomic operations tool %f s \n", runs, atomic_mean_time);
  printf("Using atomic operations also produced an output of %d \n", h_shared_var_atomic);

  printf("In Total, %d runs using unsafe operations tool %f s \n", runs, unsafe_mean_time);
  printf("Using atomic operations also produced an output of %d \n", h_shared_var_unsafe);

  print_read_results(h_values_read, d_values_read_atomic, N, "atomic");
  print_read_results(h_values_read, d_values_read_unsafe, N, "unsafe");

  cudaFree(d_shared_var);
  cudaFree(d_values_read_atomic);
  cudaFree(d_values_read_unsafe);
  free(h_values_read);
  cudaDeviceReset();
}
int main(int argc, char**argv) {
  int nDeviceNumber = 0;
  // 检测当前设备与cuda兼容的设备
  ErrorCheck(cudaGetDeviceCount(&nDeviceNumber));
  // set up device
  int dev = 0;
  // 设置GPU设备
  ErrorCheck(cudaSetDevice(dev));

  
  return 0;
}