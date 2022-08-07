#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // comment the line below to test without stride
  // int stride = blockDim.x * gridDim.x;
  // This is for loop is for CPU-only applications
  // for(int i = 0; i < N; ++i)
  // {
  //   result[i] = a[i] + b[i];
  // }
  // Without stride
  if (idx < N)
  {
    result[idx] = a[idx] + b[idx];
  }
  // With stride
  // for(int i = idx; i < N; i += stride)
  // {
  //   result[idx] = a[idx] + b[idx];
  // }
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main()
{
  const int N = 2<<20;  // left shift operation gives: 2097152 as decimal
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  // The following is for CPU-only applications
  // a = (float *)malloc(size);
  // b = (float *)malloc(size);
  // c = (float *)malloc(size);

  // Accelerated version
  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  // This function is executed by the CPU
  // addVectorsInto(c, a, b, N);
 
  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 256;
  numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  printf("The number of blocks is: %d\n", numberOfBlocks);

  // Accelerated version uses a kernel instead
  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  checkCuda( cudaGetLastError() );
  checkCuda( cudaDeviceSynchronize() );

  checkElementsAre(7, c, N);

  // Free of memory for CPU-nly applications
  // free(a)
  // free(b)
  // free(c)

  // Free Memory - Accelerated version
  checkCuda( cudaFree(a) );
  checkCuda( cudaFree(b) );
  checkCuda( cudaFree(c) );
}
