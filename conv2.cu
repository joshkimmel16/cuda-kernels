#include <iostream>
#include <stdio.h>
#include <math.h>

#define kx 3
#define ky 3
#define nx 14
#define ny 14
#define ni 512
#define nn 512

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

// CURRENT MEMORY PERFORMANCE = 14.63 GB/s

// perform a single application (matrix-vector multiply) of 1 weights matrix to a full single input feature map
// this means that the batch size is 1(?)
// the dimensions of the weights matrix are (kx, ky)
// the dimensions of all input and output feature maps are (nx, ny)
// the number of input feature maps is ni
// the number of output feature maps is nn
// the input and output feature maps are thus represented as 3D arrays (logically)
// the corresponding weights matrices are thus represented as a 4D array (logically)
// this is what is done in a 3D convolution layer
// this method utilizes a scratchpad memory for better thread block performance
__global__
void matrix_vector_mult(int *inp, int *outp, int *kern)
{
    // scratchpad memory used for shared variables
    // NOTE: can now hold both entire feature map and entire weights matrix in shared memory
    __shared__ int temp_inp[nx * ny]; // input matrix
    __shared__ int temp_kern[kx * ky]; // kernel matrix

    // only 1 thread in block needs to populate all shared variables but temp_ind
    if (threadIdx.x == 0) {
      int hold = kx * ky;
      int k_start = blockIdx.x * kx * ky;
      for (int j = 0; j < hold; j++) { // populate temp_kern
        int t = k_start + j;
        temp_kern[j] = kern[t];
      }
    }

    int i_index = ((blockIdx.x / nn) * nx * ny) + threadIdx.x; // 1 input feature map per nn output feature maps 
    int n_index = ((blockIdx.x % nn) * nx * ny) + threadIdx.x; // rotate through output feature maps constantly

    temp_inp[threadIdx.x] = inp[i_index]; // piecemeal load in the input feature map
    
    __syncthreads(); // sync all threads to this point - input feature map loaded 

    int out = 0;
    int l_start = threadIdx.x - ky/2 - (ny * (kx/2));
    for (int i=0; i<kx; i++) {
       for (int j=0; j<ky; j++) {
          int curr = l_start + (ny*i) + j;
          int k_index = (i*ky) + j;
          if ((curr >= 0) && (curr <= (nx*ny-1))) { // check against barriers of input feature map
            out += temp_inp[curr] * temp_kern[k_index];
          }
       }
    }
    outp[n_index] += out;
}

int main(void)
{
    // declare host + device pointers
    int *inp, *outp, *kern;
    int *d_inp, *d_outp, *d_kern;
    
    // compute array sizes
    int i_size = ni*nx*ny;
    int o_size = nn*nx*ny;
    int k_size = nn*ni*kx*ky;
    
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
    // # blocks = # of distinct weights matrices
    // # threads / block = # of elements in a single input/output feature map
    matrix_vector_mult<<<ni*nn, nx*ny>>>(d_inp, d_outp, d_kern);
    
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