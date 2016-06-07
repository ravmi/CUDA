#ifndef REVERSE_CUH
#define REVERSE_CUH
#include<stdlib.h>
#define GRID_SIZE_LIMIT 60000
template<typename T>
__global__ void kreverse(T* ptr, unsigned n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n/2); i+= gridDim.x * 
	    blockDim.x) {
	T tmp = ptr[i];
	ptr[i] = ptr[n - 1 - i];
        ptr[n - 1 - i] = tmp;
    }
}

template<typename T>
cudaError reverse(T* ptr, unsigned n) {
    int block_size = 1024;
    int grid_size = min(GRID_SIZE_LIMIT, (n + block_size - 1)/block_size);
    kreverse<<<grid_size, block_size>>>(ptr, n);
    return cudaGetLastError();
}
#endif
