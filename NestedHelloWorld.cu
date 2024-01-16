#include "common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void nestedHelloWorld(int const iSize, int iDepth) 
{
  int tid = threadIdx.x;
  printf("Recursion=%d: Hello World from thread %d block %d \n", iDepth, tid, blockIdx.x);

  if (iSize == 1) return;

  int nthreads = iSize >> 1;

  if (tid == 0 && nthreads > 0) {
    nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);
    printf("---->nested execution depth: %d\n", iDepth);
  }
}
// nvcc -arch=sm_35 -rdc=true NestedHelloWorld.cu -o NestedHelloWorld

int main(int argc, char**aargv) 
{
  int size = 8;
  int blocksize = 8;
  int igrid = 1;

  if (argc > 1) {
    igrid = atoi(argv[1]);
    size = igrid * blocksize;
  }

  dim3 block(blocksize, 1);
  dim3 grid((size - 1) / block.x + 1, 1);
  printf("%s execution configuration: grid %d block %d\n", argv[0], grid.x, block.x);
  nestedHelloWorld<<<grid, block>>>(block.x, 0);
  cudaDeviceReset();
  return 0;
}