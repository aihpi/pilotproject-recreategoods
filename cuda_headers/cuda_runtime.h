#ifndef CUDA_RUNTIME_H
#define CUDA_RUNTIME_H

#include <stdlib.h>
#include <math.h>

// Basic CUDA types
typedef enum {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1
} cudaError_t;

// Include vector types
#include "vector_types.h"

// CUDA device management
cudaError_t cudaGetLastError();
const char* cudaGetErrorString(cudaError_t error);

// Memory management
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, int kind);

// CUDA kernel launch parameters
#define __global__ 
#define __device__
#define __host__
#define __shared__

// CUDA kernel launch syntax
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
};

// CUDA thread/block indices
struct uint3 {
    unsigned int x, y, z;
};

// Thread/block index variables
extern __device__ uint3 threadIdx;
extern __device__ uint3 blockIdx;
extern __device__ dim3 blockDim;
extern __device__ dim3 gridDim;

// CUDA memory copy kinds
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2
#define cudaMemcpyDeviceToDevice 3

// CUDA device synchronization
cudaError_t cudaDeviceSynchronize();

// Add these functions to support kernel launch syntax (<<<grid, block>>>)
extern "C" {
    cudaError_t __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, void* stream = 0);
    cudaError_t __cudaPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem, void* stream);
}

// Include helper_math.h after our definitions
#include "helper_math.h"

#endif // CUDA_RUNTIME_H 