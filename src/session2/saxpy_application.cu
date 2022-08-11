#include <stdio.h>
// #define N 2048 * 2048 // Number of elements in each vector

/*
 * Optimize this already-accelerated codebase. Work iteratively,
 * and use nsys to support your work.
 *
 * Aim to profile `saxpy` (without modifying `N`) running under
 * 20us.
 *
 * Some bugs have been placed in this codebase for your edification.
 */

// __global__
// void saxpy(int * a, int * b, int * c)

__global__
void saxpy (float * a, float * b, float * c, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i=tid; i<N; i+=stride)
  {
    c[i] = 2 * a[i] + b[i];
  }
}
// {
//   // int tid = blockIdx.x * blockDim.x * threadIdx.x;
//   int tid = blockIdx.x * blockDim.x + threadIdx.x;
//   int stride = blockDim.x * gridDim.x;
//
//   // if ( tid < N )
//   //   c[tid] = 2 * a[tid] + b[tid];
//   for(int i=tid; i<N; i+=stride)
//   {
//     c[i] = 2 * a[i] + b[i];
//   }
// }

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

int main()
{
  const int N = 2048 * 2048;
  float *a, *b, *c;

  int size = N * sizeof (int); // The total number of bytes per vector

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  int deviceId;
  cudaGetDevice(&deviceId);

  // Prefetching the vectors to the GPU
  cudaMemPrefetchAsync(a, size, deviceId);
  cudaMemPrefetchAsync(b, size, deviceId);
  cudaMemPrefetchAsync(c, size, deviceId);

  // Initialize memory
  // for( int i = 0; i < N; ++i )
  // {
  //     a[i] = 2;
  //     b[i] = 1;
  //     c[i] = 0;
  // }

  size_t threads_per_block;
  size_t number_of_blocks;

  threads_per_block = 128;
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);
  int SMs = props.multiProcessorCount;

  number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
  number_of_blocks = number_of_blocks + (SMs - number_of_blocks % SMs);

  // printf("------------------------------");
  // printf("\nVector length is: %d", N);
  // printf("\nNumber of threads per block is: %d", threads_per_block);
  // printf("\nNumber of blocks is: %d\n", number_of_blocks);

  initWith<<< number_of_blocks, threads_per_block >>> (2, a, N);
  initWith<<< number_of_blocks, threads_per_block >>> (1, b, N);
  initWith<<< number_of_blocks, threads_per_block >>> (0, c, N);

  // saxpy <<< number_of_blocks, threads_per_block >>> ( a, b, c );
  saxpy <<< number_of_blocks, threads_per_block >>> ( a, b, c, N );

  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("\nError: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("\nError: %s\n", cudaGetErrorString(asyncErr));

  // Prefetching the output vector back to CPU. `cudaCpudeviceId` is a built-in CUDA variable
  // cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);  // this increases time 

  // Print out the first and last 5 values of c for a quality check
  for( int i = 0; i < 5; ++i )
    printf("c[%d] = %f, ", i, c[i]);
  printf ("\n");
  for( int i = N-5; i < N; ++i )
    printf("c[%d] = %f, ", i, c[i]);
  printf ("\n");

  cudaFree( a );
  cudaFree( b );
  cudaFree( c );
}
