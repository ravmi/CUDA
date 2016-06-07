#define HAMMING
#ifdef HAMMING
#include<stdio.h>
#include<stdint.h>
#include<stdarg.h>
#include<iostream>
#include<string>

#define BLOCK_SIZE 16
#define DEPTH (16)


//n musi być wielokrotnością sizeof(type) * 8 * BLOCK_SIZE

#define CUDA_CHECK_RETURN(value) { \
    cudaError_t _m_cudaStat = value; \
    if (_m_cudaStat != cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
        exit(1); \
    } }
//#define printf(asdf, ...) ;


typedef uint64_t type;
/*
   __device__ print_length(uint32_t *ar, int length) {
   for(int i = 0; i < length; i++)
 */
__global__ void ghamming(type *in, int k, int n, int2* out, int* last, int max, int threshold) {
    __shared__ type prefetch[2][BLOCK_SIZE][DEPTH+1];
    int m = n/sizeof(type)/8; // how many uint32_t are there in a vector
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int bX = blockIdx.x * blockDim.x;
    int bY = blockIdx.y * blockDim.y;
    //   if(x==0 && y==0)
    //  printf("m: %d\n", m);

    // for(int i = 0; i < m; i++) {
    //  printf("Watek (y, x):(%d, %d): porownujemy: %d, %d \n",y, x, in[m* x + i], in[y*m + i] );
    //   }
    int pc = 0;
#pragma unroll 4
    for(int j = 0; j < m; j+= DEPTH) {
        __syncthreads();
            prefetch[0][threadIdx.y][threadIdx.x] = in[(bX + threadIdx.y) * m + threadIdx.x + j];
            prefetch[1][threadIdx.y][threadIdx.x] = in[(bY + threadIdx.y) * m + threadIdx.x + j];
        //      printf("Watek (y, x):(%d, %d):\n pf[0][%d][%d] = %d\n pf[1][%d][%d] = %d\n", y, x, threadIdx.y, i, prefetch[0][threadIdx.y][i], threadIdx.y, i, prefetch[1][threadIdx.y][i]);
        __syncthreads();
        if(x > y && x < k && y < k) {
            for(int i = 0; i <  DEPTH; i++) {
                pc += __popcll(prefetch[0][threadIdx.x][i] ^ prefetch[1][threadIdx.y][i]);
                //              printf("Watek (y, x):(%d, %d): pc zwiekszono o = %d podczas porowyanania: %d, %d\n", y, x, __popc(prefetch[0][threadIdx.x][i] ^ prefetch[1][threadIdx.y][i]), prefetch[0][threadIdx.x][i],                     prefetch[1][threadIdx.y][i]);
            }

        }

    }
    if(x > y && x < k && y < k) {
        //printf("%d %d: %d\n", y, x, pc);
        if(pc < threshold) {
            int q = atomicInc((unsigned*) last, (unsigned) max);
            out[q].x = x;
            out[q].y = y;
            //          printf("zapisueje Watek (y, x):(%d, %d): out[%d].x = %d, out[%d].y = %d\n", y, x, out[q].x, x, out[q].y, y);
        }
    }



}

void hamming(void *in, int k, int n, int2* out, int* last, int max, int threshold) {
    int zero = 0;
    CUDA_CHECK_RETURN(cudaMemcpy(last, &zero, sizeof(int), cudaMemcpyHostToDevice));
    // printf("starting hamming: \n");
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid((k + BLOCK_SIZE - 1)/BLOCK_SIZE, (k + BLOCK_SIZE - 1)/BLOCK_SIZE, 1);
    ghamming<<<grid, block>>>((type*)in, k, n, out, last, max, threshold);
    CUDA_CHECK_RETURN(cudaGetLastError());
}

#endif

