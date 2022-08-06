#include <stdio.h>

void helloCPU()
{
  printf("Hello from the CPU.\n");
}

/*
 * Refactor the `helloGPU` definition to be a kernel
 * that can be launched on the GPU. Update its message
 * to read "Hello from the GPU!"
 */

__global__ void helloGPU()
{
  printf("Hello from the GPU.\n");
}

int main()
{

  helloCPU();

  /*
   * Refactor this call to `helloGPU` so that it launches
   * as a kernel on the GPU.
   */

  helloGPU<<<1, 1>>>();

  /*
   * Add code below to synchronize on the completion of the
   * `helloGPU` kernel completion before continuing the CPU
   * thread.
   */
   cudaDeviceSynchronize();
}

/*
  The command to compile is as follows:
  nvcc -arch=sm_70 -o hello-gpu 01-hello-gpu.cu -run
*/

// NOTES
// Error Messages when
// __global__ keyword is ommited when defining the kernel
//
// 01-hello/01-hello-gpu.cu(29): error: a host function call cannot be configured
// 1 error detected in the compilation of "/tmp/tmpxft_00000077_00000000-8_01-hello-gpu.cpp1.ii".
// HOST function call means that the CPU cannot call the GPU function

// If cudaDeviceSynchronize() is removed
// The CPU finished the main function without waiting fot the GPU to return

