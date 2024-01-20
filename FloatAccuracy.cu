#include "common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void kernel(float *F, double *D) {
  float var1 = *F;
  float var2 = *D;
  printf("device single precision representation is %.2f\n", var1);
  printf("device double precision representation is %.2f\n", var2);
}

int main(int argc, char**argv) {

  float hostF = 0.0;
  double hostD = 0.0;

  if (argc == 2) {
    hostF = (float) atof(argv[1]);
    hostD = (double) atof(aargv[1]);
  } else {
    printf("input a float number\n");
    return -1;
  }

  int nDeviceNumber = 0;
  // 检测当前设备与cuda兼容的设备
  ErrorCheck(cudaGetDeviceCount(&nDeviceNumber));
  // set up device
  int dev = 0;
  // 设置GPU设备
  ErrorCheck(cudaSetDevice(dev));

  float *deviceF;
  double* deviceD;

  cudaMalloc((void**)&deviceF, sizeof(float));
  cudaMalloc((void**)&deviceD, sizeof(double));
  cudaMemcpy(deviceF, &hostF, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceD, &hostD, sizeof(double), cudaMemcpyHostToDevice);

  printf("host single precision representation is %.2f\n", hostF);
  printf("host double precision representation is %.2f\n", hostD);

  kernel<<<1,1>>>(deviceF, deviceD);
  cudaDeviceSynchronize();
  cudaFree(deviceF);
  cudaFree(deviceD);
  cudaDeviceReset();
  return 0;
}