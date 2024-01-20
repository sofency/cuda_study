#include <cuda_runtime.h>
#include <stdio.h>
#include "common/common.h"

// 32位比特位全部都是1， 每个比特位代表一个线程 全为1　表示所有线程都执行
const unsigned FULL_MASK = 0xffffffff;

__global__ void shuffle_broadcast(int *in, int* out, int srcLane)
{
  int value = in[threadIdx.x];
  // T 是要交换的数据类型，mask 是用于同步的掩码，var 是要交换的值，srcLane 是源线程的标识，width 是线程束的宽度，默认为 warpSize。
  // T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);

  // 将value(参数)从其他线程传递出来 给value(左边)赋值
  value = __shfl_sync(FULL_MASK, value, srcLane, 32);
  out[threadIdx.x] = value;
}

/**
 * @brief 
 * int __shfl_up(int var, unsigned int delta, int width=wrapSize)
 * 从与当前线程向后偏移delta的线程读取var值   width是2的n次方 
 * 待测试 0 1 2 0 1 2 3 4 5

 * __shfl_xor(int var, int lanMask, int width=wrapSize) 将当前线程的id与lanMask做异或操作
 * @return __global__ 
 */
__global__ void shuffle_up_demo(int *in, int* out, int srcLane)
{
  int value = in[threadIdx.x];
  value = __shfl_up_sync(FULL_MASK, value, srcLane); // w最后一个参数 默认为32
  // value = __shfl_down_sync(FULL_MASK, value, srcLane); 3 4 5 ... 29 30 31 29 30 31
  // value = __shfl_xor_sync(FULL_MASK, value, srcLane, 32); 0 1 2 3 进行异或处理 1 0 3 2 
  out[threadIdx.x] = value;
}




int main(int argc, char**argv)
{
  int nDeviceNumber = 0;
  // 检查是否支持cuda
  ErrorCheck(cudaGetDeviceCount(&nDeviceNumber));
  
  int dev = 0;
  // 设置使用哪个GPU
  ErrorCheck(cudaSetDevice(dev));

  int nElem = 32;
  // 分配主机内存 0拷贝方式
  int *in = NULL;
  int *out = NULL;
  ErrorCheck(cudaHostAlloc((void**)&in, sizeof(int) * nElem, cudaHostAllocDefault));
  ErrorCheck(cudaHostAlloc((void**)&out, sizeof(int) * nElem, cudaHostAllocDefault));

  for (int i = 0; i< nElem; i++) {
    in[i] = i;
  }

  dim3 block(nElem);
  dim3 grid(1);
  shuffle_broadcast<<<grid, block>>>(in, out, 3);
  for (int i = 0; i< nElem; i++) {
    printf("out element is, id:%d, value:%d\n", i, out[i]);
  }

  cudaFreeHost(in);
  cudaFreeHost(out);
  cudaDeviceReset();
  return 0;

}