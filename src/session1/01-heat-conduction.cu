#include <stdio.h>
#include <math.h>
#include <assert.h>

// Simple define to index into a 1D array from 2D space
#define I2D(num, c, r) ((r)*(num)+(c))

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

/*
 * `step_kernel_mod` is currently a direct copy of the CPU reference solution
 * `step_kernel_ref` below. Accelerate it to run as a CUDA kernel.
 */

__global__
void step_kernel_mod(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
  int i00, im10, ip10, i0m1, i0p1;
  float d2tdx2, d2tdy2;

  // Mapping threads to 2D silver plate coordinates (matrix)
  // Changing of column is changing in the x direction
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  // Changing of row is changing in the y direction
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  // loop over all points in domain (except boundary)
  if (row > 0 && col > 0 && row < ni-1 && col < nj-1)
  {
    i00 = I2D(ni, row, col);
    im10 = I2D(ni, row-1, col);
    ip10 = I2D(ni, row+1, col);
    i0m1 = I2D(ni, row, col-1);
    i0p1 = I2D(ni, row, col+1);
    // evaluate derivatives
    d2tdx2 = temp_in[im10]-2*temp_in[i00]+temp_in[ip10];
    d2tdy2 = temp_in[i0m1]-2*temp_in[i00]+temp_in[i0p1];
    // update temperatures
    temp_out[i00] = temp_in[i00]+fact*(d2tdx2 + d2tdy2);
  }
}

void step_kernel_ref(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
  int i00, im10, ip10, i0m1, i0p1;
  float d2tdx2, d2tdy2;


  // loop over all points in domain (except boundary)
  for ( int j=1; j < nj-1; j++ ) {
    for ( int i=1; i < ni-1; i++ ) {
      // find indices into linear memory
      // for central point and neighbours
      i00 = I2D(ni, i, j);
      im10 = I2D(ni, i-1, j);
      ip10 = I2D(ni, i+1, j);
      i0m1 = I2D(ni, i, j-1);
      i0p1 = I2D(ni, i, j+1);

      // evaluate derivatives
      d2tdx2 = temp_in[im10]-2*temp_in[i00]+temp_in[ip10];
      d2tdy2 = temp_in[i0m1]-2*temp_in[i00]+temp_in[i0p1];

      // update temperatures
      temp_out[i00] = temp_in[i00]+fact*(d2tdx2 + d2tdy2);
    }
  }
}

int main()
{
  int istep;
  int nstep = 200; // number of time steps

  // Specify our 2D dimensions
  const int ni = 200;
  const int nj = 100;
  float tfac = 8.418e-5; // thermal diffusivity of silver

  float *temp1_ref, *temp2_ref, *temp1, *temp2, *temp_tmp;

  const int size = ni * nj * sizeof(float);

  // CPU-only allocation
  temp1_ref = (float*)malloc(size);
  temp2_ref = (float*)malloc(size);
  // temp1 = (float*)malloc(size);
  // temp2 = (float*)malloc(size);

  // Allocation of memory for both CPU and GPU
  cudaMallocManaged(&temp1, size);
  cudaMallocManaged(&temp2, size);

  // Initialize with random data
  for( int i = 0; i < ni*nj; ++i) {
    temp1_ref[i] = temp2_ref[i] = temp1[i] = temp2[i] = (float)rand()/(float)(RAND_MAX/100.0f);
  }

  // Execute the CPU-only reference version
  for (istep=0; istep < nstep; istep++) {
    step_kernel_ref(ni, nj, tfac, temp1_ref, temp2_ref);

    // swap the temperature pointers so after an iteration the pointers
    // point to the new temperatures
    temp_tmp = temp1_ref;
    temp1_ref = temp2_ref;
    temp2_ref= temp_tmp;
  }

  dim3 threads_per_block(20, 10, 1);
  // Better definition is given by multiples of 2: 
  // dim3 threads_per_block(32, 16, 1);
  dim3 number_of_blocks(nj / threads_per_block.x + 1, ni / threads_per_block.y + 1, 1);

  // Execute the modified version using same data
  for (istep=0; istep < nstep; istep++) {
    step_kernel_mod <<< number_of_blocks, threads_per_block >>> (ni, nj, tfac, temp1, temp2);

    // swap the temperature pointers so after an iteration the pointers
    // point to the new temperatures
    temp_tmp = temp1;
    temp1 = temp2;
    temp2= temp_tmp;
  }

  cudaError_t ierrSync, ierrAsync;

  ierrSync = checkCuda( cudaGetLastError() );
  if (ierrSync != cudaSuccess) {printf("The above error was a Sync error\n"); }
  ierrAsync = checkCuda( cudaDeviceSynchronize() );
  if (ierrAsync != cudaSuccess) {printf("The above error was an Async error\n"); }

  float maxError = 0;
  // Output should always be stored in the temp1 and temp1_ref at this point
  for( int i = 0; i < ni*nj; ++i ) {
    if (abs(temp1[i]-temp1_ref[i]) > maxError) { maxError = abs(temp1[i]-temp1_ref[i]); }
  }

  // Check and see if our maxError is greater than an error bound
  if (maxError > 0.0005f)
    printf("Problem! The Max Error of %.5f is NOT within acceptable bounds.\n", maxError);
  else
    printf("The Max Error of %.5f is within acceptable bounds.\n", maxError);

  // Free Memory for CPU allocations
  free( temp1_ref );
  free( temp2_ref );

  // Free Memory for GPU allocations
  cudaFree( temp1 );
  cudaFree( temp2 );

  return 0;
}
