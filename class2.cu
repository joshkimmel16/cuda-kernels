#include <iostream>
#include <stdio.h>
#include <math.h>

#define ni 4096
#define nn 1024

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void random_ints(int* a, int N)
{
   int i;
   for (i = 0; i < N; i++) 
   {
      a[i] = rand(); 
   }
}

void zeros(int* a, int N)
{
   int i;
   for (i = 0; i < N; i++) 
   {
      a[i] = 0; 
   }
}

// CURRENT MEMORY PERFORMANCE = 14.23 GB/s

// perform 1 column of the matrix-vector multiply (1 input, 1 weights vector)
// this means that the batch size is 1(?)
// the dimensions of the weights matrix are (ni, nn) => 2D array
// the full input is a vector of dimension ni (represented as an array)
// the full output is a vector of dimension nn (represented as an array)
// this is what is done in a fully-connected classifier layer
// this method utilizes a scratchpad memory for better thread block performance
__global__
void matrix_vector_mult(int *inp, int *outp, int *kern)
{
    // scratchpad memory used for shared variables
    __shared__ int temp_kern[nn]; // weights vector
    __shared__ int temp_inp; // single input vector value

    // populate shared data structures
    // only 1 thread needs to handle everything
    if (threadIdx.x == 0) {
      for (int i=0; i<nn; i++) {
         int k_index = blockIdx.x + i*ni; // weights matrix is row-major
         temp_kern[i] = kern[k_index]; 
      }
      temp_inp = inp[blockIdx.x];
    }
    
    __syncthreads(); // sync all threads to this point 

    // populate output
    outp[threadIdx.x] += (temp_inp * temp_kern[threadIdx.x]);
}

int main(void)
{
    // declare host + device pointers
    int *inp, *outp, *kern;
    int *d_inp, *d_outp, *d_kern;
    
    // compute array sizes
    int i_size = ni;
    int o_size = nn;
    int k_size = nn*ni;
    
    // allocate space for each array on the device
    gpuErrchk( cudaMalloc(&d_inp, i_size*sizeof(int)) );
    gpuErrchk( cudaMalloc(&d_outp, o_size*sizeof(int)) );
    gpuErrchk( cudaMalloc(&d_kern, k_size*sizeof(int)) );
    
    // allocate space and populate each array on the host
    inp = (int*)malloc(i_size*sizeof(int)); 
    outp = (int*)malloc(o_size*sizeof(int));
    kern = (int*)malloc(k_size*sizeof(int)); 
    random_ints(inp, i_size);
    zeros(outp, o_size);
    random_ints(kern, k_size);
    
    // copy populated host arrays to corresponding device arrays
    gpuErrchk( cudaMemcpy(d_inp, inp, i_size*sizeof(int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_outp, outp, o_size*sizeof(int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_kern, kern, k_size*sizeof(int), cudaMemcpyHostToDevice) );

    // launch all threads on device
    // # blocks = # of columns (ni)
    // # threads / block = # of rows (nn)
    matrix_vector_mult<<<ni, nn>>>(d_inp, d_outp, d_kern);
    
    // determine if run succeeded
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    // copy output array back to host
    gpuErrchk( cudaMemcpy(outp, d_outp, o_size, cudaMemcpyDeviceToHost) );

    // free all memory
    free(inp); free(outp); free(kern);
    gpuErrchk( cudaFree(d_inp) ); gpuErrchk( cudaFree(d_outp) ); gpuErrchk( cudaFree(d_kern) );

    return 0;
}