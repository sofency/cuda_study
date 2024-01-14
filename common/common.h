#include <sys/time.h>
#include <cuda_runtime.h>
#include <stdio.h>

cudaError_t ErrorCheck(cudaError_t status, const char *filename, int lineNumber)
{
  if (status != cudaSuccess)
  {
    printf("CUDA API error:\r\n code=%d, name=%s, description=%s\r\n
    file=%s, line=%d\r\n", status, cudaGetErrorName(status), cudaGetErrorString(status), filename, lineNumber);
  }
  return status;
}

inline double GetCPUSecond()
{
  struct timeval tp;
  struct timezone tzp;
  int i = gettimeofday(&tp, &tzp);
  // 秒数+微秒*1.e-6 就是秒数
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}