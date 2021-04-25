#include <iostream>
#include <stdio.h>
#include <math.h>

#define kx 3
#define ky 3
#define nx 224
#define ny 224
#define ni 64
#define nn 64

#define batch 64

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

// CURRENT MEMORY PERFORMANCE = 5.77 GB/s

// perform a single application (matrix-vector multiply) of 1 weights matrix to a subset of a single input feature map
// the batch size (batch) determines the size of the subset
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
    // NOTE: must batch enough such that this data can fit in the shared memory
    __shared__ int temp_kern[nn * kx * ky]; // all kernel matrices for given input feature map
    __shared__ int temp_inp[nx * ny / batch]; // batched subset of given input feature map

    // only 1 thread in block needs to populate all shared variables but temp_ind
    if (threadIdx.x == 0) {
      int hold = nn* kx * ky;
      int k_start = (blockIdx.x/batch) * kx * ky; // every (batch) thread blocks use the same weights matrices
      for (int j = 0; j < hold; j++) { // populate temp_kern
        int t = k_start + j;
        temp_kern[j] = kern[t];
      }
    }

    int i_index = (blockIdx.x * (nx * ny / batch)) + threadIdx.x; // 1 thread block per subset of each feature map
    temp_inp[threadIdx.x] = inp[i_index]; // piecemeal load in the input feature map
    
    __syncthreads(); // sync all threads to this point - input feature map loaded

    int l_start = threadIdx.x - ky/2 - (ny/(batch/2) * (kx/2));
    for (int i=0; i<nn; i++) {
        int out = 0;
        for (int j=0; j<kx; j++) {
          for (int k=0; k<ky; k++) {
            int curr = l_start + (ny/(batch/2)*j) + k;
            int k_index = (i*kx*ky) + (j*ky) + k;
            if ((curr >= 0) && (curr <= (nx*ny/batch-1))) { // check against barriers of input feature map
              out += temp_inp[curr] * temp_kern[k_index];
            }
          }
        }
        // store output
        int n_index = (i * nx * ny) + threadIdx.x; // rotate through output feature maps constantly
        outp[n_index] += out;
    }
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
    // # blocks = # input feature maps * # batches / input feature map
    // # threads / block = # elements in each batch
    matrix_vector_mult<<<ni*batch, nx*ny/batch>>>(d_inp, d_outp, d_kern);

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