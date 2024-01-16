#include<stdio.h>

// grid >> block >> Thread
// 每个线程是由blockIdx和threadIdx唯一标识
// 线程标识是cuda平台内置和分配，并且可以在内核程序中访问
// 线程标识是三维向量，通过下标x,y,z访问
// 共享内存 
// 内核函数中由 __shared__修饰的变量都保存在共享内存中，共享是片上存储空间，具有低延迟和高带宽特点
// 常量内存 由__constant__修饰的变量存放在常量内存中，常量内存可以被所有内核代码访问 


// __host__ 运行在主机(CPU)  __global__  运行在设备 必须返回void __device__ 运行在设备

// 内核函数被GPU上线程并发执行（硬件层面并发）
// 内核函数只能访问GPU内存，必须返回void,不能使用变长参数， 不能使用静态变量，不能使用函数指针，内核函数具有异步性

// 如果block维度是(Dx， Dy), 线程标识为(i,j) thread ID = (i+j*Dx) 
__global__ void helloWorldGPU()
{
  printf("blockDim:x=%d, y=%d, z=%d ", blockDim.x, blockDim.y, blockDim.z);
  printf("gridDim:x=%d, y=%d, z=%d ", gridDim.x, gridDim.y, gridDim.z);
  printf("threadIdx:x=%d, y=%d, z=%d \n", threadIdx.x, threadIdx.y, threadIdx.z);
}

// nvcc HelloWorld.cu --output-file HelloWorld
// ./HelloWorld
__device__ float factor = 10.0;
__global__ void allocateMemory(float*d_A)
{
  printf("给d_A赋值 %f\n", factor);
  *d_A = factor;
}

int main(int argc, char **argv) 
{
  printf("Hello World from CPU\n");
  // grid 表示线程的网格 block表示线程的框
  
  dim3 grid;
  grid.x = 2;
  grid.y = 2;

  dim3 block;
  block.x = 2;
  block.y = 2;
  helloWorldGPU<<<grid, block>>>();
  cudaDeviceSynchronize();

  float*d_A;
  cudaMalloc((void**)&d_A, sizeof(float));
  allocateMemory<<<1,1>>>(d_A);
  cudaDeviceSynchronize();


  float*h_A;
  // cudaMemcpy(&h_A, d_A, sizeof(float), cudaMemcpyDeviceToHost);
  ErrorCheck(cudaMemcpyFromSymbol(h_A, d_A, sizeof(float))); //使用上面 或者这个

  printf("copy from GPU value %f\n", h_A);

  cudaDeviceReset();
  return 0;
}
