#include <stdio.h>

__global__ void printSuccessForCorrectExecutionConfiguration()
{

  // if(threadIdx.x == 1023 && blockIdx.x == 255)
  if(threadIdx.x == 0 && blockIdx.x == 0)
  {
    printf("Success!\n");
  } else {
    printf("Failure. Update the execution configuration as necessary.\n");
  }
}

int main()
{
  /*
   * Update the execution configuration so that the kernel
   * will print `"Success!"`.
   */

  printSuccessForCorrectExecutionConfiguration<<<1, 1>>>();
  cudaDeviceSynchronize();
}

/*
  ALTERNATIVE SOLUTION
*/

/*

#include <stdio.h>

__global__ void printSuccessForCorrectExecutionConfiguration()
{

  if(threadIdx.x == 1023 && blockIdx.x == 255)
  {
    printf("Success!\n");
  } else {
    printf("Failure. Update the execution configuration as necessary.\n");
  }
}

int main()
{

  printSuccessForCorrectExecutionConfiguration<<<256, 1024>>>();
  cudaDeviceSynchronize();
}

*/
