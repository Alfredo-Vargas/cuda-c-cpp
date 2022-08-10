__global__
void deviceKernel(int *a, int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] = 1;
  }
}

void hostFunction(int *a, int N)
{
  for (int i = 0; i < N; ++i)
  {
    a[i] = 1;
  }
}

int main()
{

  int N = 2<<24;
  size_t size = N * sizeof(int);
  int *a;
  cudaMallocManaged(&a, size);

  /*
   * Conduct experiments to learn more about the behavior of
   * `cudaMallocManaged`.
   *
   * What happens when unified memory is accessed only by the CPU?
   * What happens when unified memory is accessed only by the GPU?
   * What happens when unified memory is accessed first by the GPU then the CPU?
   * What happens when unified memory is accessed first by the CPU then the GPU?
   *
   * Hypothesize about UM behavior, page faulting specificially, before each
   * experiment, and then verify by running `nsys`.
   */

  // Evidence of Memory Migration or Page Faulting when unified memory 
  // is accessed only by the CPU
  // hostFunction(a, N);

  // Evidence of Memory Migration or Page Faulting when unified memory 
  // is accessed only by the GPU
  // size_t threadsperBlock;
  // size_t numberOfBlocks;
  // threadsperBlock = 256;
  // numberOfBlocks = (N + threadsperBlock - 1) / threadsperBlock;
  // deviceKernel<<<numberOfBlocks, threadsperBlock>>>(a, N);
  // cudaDeviceSynchronize();

  // Evidence of Memory Migration or Page Faulting when unified memory
  // is accessed first by the CPU and then by the GPU
  // hostFunction(a, N);
  //
  // size_t threadsperBlock;
  // size_t numberOfBlocks;
  // threadsperBlock = 256;
  // numberOfBlocks = (N + threadsperBlock - 1) / threadsperBlock;
  // deviceKernel<<<numberOfBlocks, threadsperBlock>>>(a, N);
  // cudaDeviceSynchronize();

  // Evidence of Memory Migration or Page Faulting when unified memory
  // is accessed first by the GPU and then by the CPU
  size_t threadsperBlock;
  size_t numberOfBlocks;
  threadsperBlock = 256;
  numberOfBlocks = (N + threadsperBlock - 1) / threadsperBlock;
  deviceKernel<<<numberOfBlocks, threadsperBlock>>>(a, N);
  cudaDeviceSynchronize();

  hostFunction(a, N);

  cudaFree(a);
}
