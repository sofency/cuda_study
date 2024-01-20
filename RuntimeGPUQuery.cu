#include<common/common.h>
#include<cuda_runtime.h>
#include<stdio.h>

/**
 * @brief
 * 线程束: 每个线程束包含32个线程，每个线程束只能包含相同线程块中的线程
 * 每个GPU可以执行成千上万的线程 线程网格中的所有线程块被分配到CPU中的SM上执行
 * 每个线程块在同一个SM上执行，每个SM可执行多个线程块，当线程块被分配到到SM上时，会再次以32个线程为一组分割
 * @param argc 
 * @param argv 
 * @return int 
 */


int main(int argc, char **argv)
{
  printf("%s Starting...\n", argv[0]);

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0) {
    printf("Threr are no aavaialabale device(s) that support CUDA\n");
    return 1;
  } else {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }

  int dev = 0, driverVersion = 0, runtimeVersion = 0;
  cudaSetDevice(dev);

  // 查看当前cuda的属性
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  // 
  if (deviceProp.concurrentKernels) {
    printf("concurrent kernel is supported on this GPU, begin to execute kernel\n");
  } else {
    printf("concurrent kernel not supported on this GPU\n");
  }

  printf("Device %d: \"%s\" \n", dev, deviceProp.name);

  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  printf("CUDA Driver Version / Runtime Version    %d.%d / %d.%d\n", 
              driverVersion / 1000, (driverVersion % 100) / 10, 
              runtimeVersion / 1000, (runtimeVersion % 100) / 10);

  

  printf("Device name: %s\n", deviceProp.name);
  printf("Compute capaability: %d.%d\n", deviceProp.major, deviceProp.minor);
  printf("Amount of global memory: %g GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024 * 1024));
  printf("Amount of constant memory:  %g KB\n", deviceProp.totalConstMem / 1024.0);
  printf("Maximum grid size: %d %d %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
  printf("Maximum block size: %d %d %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
  printf("Number of SMs: %d\n", deviceProp.multiprocessorCount);
  printf("Maxmium amount of shared memory per block: %g KB\n", deviceProp.sharedMemPerBlock / 1024.0);
  printf("Maxmium amount of shared memory per SM: %g KB\n", deviceProp.sharedMemPerMultiprocessor / 1024.0);
  printf("Maxmium number of registers per block: %d K\n", deviceProp.regsPerBlock / 1024);
  printf("Maxmium number of registers per SM: %d K\n", deviceProp.regsPerMultiprocessor / 1024);
  printf("Maxmium number of threads per block: %d K\n", deviceProp.maxThreadsPerBlock);
  printf("Maxmium number of threads per SM: %d\n", deviceProp.maxThreaadsPerMultiProcessor);
  return 0;
}