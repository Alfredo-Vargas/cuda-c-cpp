#include <stdio.h>

__global__ void initWith(float num, float *a, int N) {

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < N; i += stride) {
    a[i] = num;
  }
}

__global__ void addVectorsInto(float *result, float *a, float *b, int N) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < N; i += stride) {
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *vector, int N) {
  for (int i = 0; i < N; i++) {
    if (vector[i] != target) {
      printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i],
             target);
      exit(1);
    }
  }
  printf("Success! All values calculated correctly.\n");
}

int main() {
  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount,
                         deviceId);

  const int N = 2 << 24;
  size_t size = N * sizeof(float);

  float *device_a;
  float *device_b;
  float *device_c;
  float *host_c;

  cudaMalloc(&device_a, size);
  cudaMalloc(&device_b, size);
  cudaMalloc(&device_c, size);
  cudaMallocHost(&host_c, size);

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 256;
  numberOfBlocks = 32 * numberOfSMs;

  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  /*
   * Create 3 streams to run initialize the 3 data vectors in parallel.
   */

  cudaStream_t stream1, stream2, stream3;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);

  /*
   * Give each `initWith` launch its own non-standard stream.
   */

  initWith<<<numberOfBlocks, threadsPerBlock, 0, stream1>>>(3, device_a, N);
  initWith<<<numberOfBlocks, threadsPerBlock, 0, stream2>>>(4, device_b, N);
  initWith<<<numberOfBlocks, threadsPerBlock, 0, stream3>>>(0, device_c, N);

  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(device_c, device_a,
                                                      device_b, N);

  cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);

  addVectorsErr = cudaGetLastError();
  if (addVectorsErr != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if (asyncErr != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(asyncErr));

  checkElementsAre(7, host_c, N);

  /*
   * Destroy streams when they are no longer needed.
   */

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaStreamDestroy(stream3);

  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
  cudaFreeHost(host_c);
}
