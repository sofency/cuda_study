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
  printf("Device %d: \"%s\" \n", dev, deviceProp.name);

  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  printf("CUDA Driver Version / Runtime Version    %d.%d / %d.%d\n", 
              driverVersion / 1000, (driverVersion % 100) / 10, 
              runtimeVersion / 1000, (runtimeVersion % 100) / 10);

  

}