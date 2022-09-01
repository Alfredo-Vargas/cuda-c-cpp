#include <stdio.h>
#include <unistd.h>

__global__ void printNumber(int number)
{
  printf("%d\n", number);
}

int main()
{
  for (int i = 0; i < 5; ++i)
  {
    // Exercise Solution: define CUDA streams
    cudaStream_t stream;          // CUDA streams are of type `cudaStream_t`
    cudaStreamCreate(&stream);    // cudaCreateStream uses a pointer of a stream
    // finish of definition of CUDA streams

    printNumber<<<1, 1, 0, stream>>>(i);

    // Exercise Solution: destruction of CUDA streams after its usage
    cudaStreamDestroy(stream);  // Note that a value, not a pointer, is passed to `cudaDestroyStream`
    // finish of the destruction of the created CUDA stream
  }
  cudaDeviceSynchronize();
}

