#include <stdio.h>

/*
 * Host function to initialize vector elements. This function
 * simply initializes each element to equal its index in the
 * vector.
 */

__global__
void initWith(float num, float *a, int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = idx; i < N; i+=stride)
  {
    a[idx] = num;
  }
}

/*
 * Device kernel stores into `result` the sum of each
 * same-indexed value of `a` and `b`.
 */

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

/*
 * Host function to confirm values in `vector`. This function
 * assumes all values are the same `target` value.
 */

void checkElementsAre(float target, float *vector, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(vector[i] != target)
    {
      printf("\nFAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
      exit(1);
    }
  }
  printf("\nSuccess! All values calculated correctly.\n");
}

int main()
{
  const int N = 2<<24;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);
  
  int deviceId;
  cudaGetDevice(&deviceId);

  // Prefetching the vectors
  cudaMemPrefetchAsync(a, size, deviceId);
  // cudaMemPrefetchAsync(b, size, deviceId);
  // cudaMemPrefetchAsync(c, size, deviceId);

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 800;

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);
  int SMs = props.multiProcessorCount;
  // The lines above can be changed by using "cudaDeviceGetAttribute(pointer_to_store, attributeName, deviceId)"
  // cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
  // printf("Device ID: %d\tNumber of SMs: %d\n", deviceId, numberOfSMs);

  numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  numberOfBlocks = numberOfBlocks + (SMs - numberOfBlocks % SMs);
  // numberOfBlocks = (((N + threadsPerBlock - 1) / threadsPerBlock) + SMs - 1) / SMs;
  printf("------------------------------");
  printf("\nVector length is: %d", N);
  printf("\nNumber of threads per block is: %d", threadsPerBlock);
  printf("\nNumber of blocks is: %d", numberOfBlocks);

  initWith<<<numberOfBlocks, threadsPerBlock>>>(3, a, N);
  initWith<<<numberOfBlocks, threadsPerBlock>>>(4, b, N);
  initWith<<<numberOfBlocks, threadsPerBlock>>>(0, c, N);


  /*
   * nsys should register performance changes when execution configuration
   * is updated.
   */


  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("\nError: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("\nError: %s\n", cudaGetErrorString(asyncErr));

  checkElementsAre(7, c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
