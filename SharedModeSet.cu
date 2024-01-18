#include <cuda_runtime.h>
#include <stdio.h>
#include "common/common.h"

int main(int argc, char**argv)
{
  int nDeviceNumber = 0;
  // 检查是否支持cuda
  ErrorCheck(cudaGetDeviceCount(&nDeviceNumber));
  
  int dev = 0;
  // 设置使用哪个GPU
  ErrorCheck(cudaSetDevice(dev));

  cudaSharedMemConfig sharedMemConfig;
  // 查询共享模式支持的字节数
  ErrorCheck(cudaDeviceGetSharedMemConfig(&sharedMemConfig));
  printf("current shared memory mode: %d\n", sharedMemConfig);

  // 修改模式
  switch (sharedMemConfig) {
    case cudaSharedMemBankSizeEightByte: 
      sharedMemConfig = cudaSharedMemBankSizeFouryte;
      ErrorCheck(cudaDeviceSetSharedMemConfig(sharedMemConfig));
      break;
    case cudaSharedMemBankSizeFouryte:
      sharedMemConfig = cudaSharedMemBankSizeEightByte;
      ErrorCheck(cudaDeviceSetSharedMemConfig(sharedMemConfig));
      break;
  }

  printf("current shared memory mode: %d\n", sharedMemConfig);

  // 配置共享内存大小
  cudaFuncCache cacheConfig;
  // 获取共享内存配置
  ErrorCheck(cudaDeviceGetCacheConfig(&cacheConfig));
  printf("default cache config for device: %d\n", cacheConfig);
  
  cacheConfig = cudaFuncCachePreferEqual; // L1缓存共享内存使用同样大小
  ErrorCheck(cudaDeviceSetCacheConfig(cacheConfig));

  cacheConfig = cudaFuncCachePreferShared; //使用比较大的共享内存
  ErrorCheck(cudaFuncSetCacheConfig(cacheConfig));

  // 获取共享内存配置
  ErrorCheck(cudaDeviceGetCacheConfig(&cacheConfig));
  printf("current cache config for device: %d\n", cacheConfig);



  cudaDeviceReset();
  return 0;
}