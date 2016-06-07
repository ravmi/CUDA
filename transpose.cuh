#ifndef TRANSPOSE_CUH
#define TRANSPOSE_CUH
#define BLOCK_DIM 32

__global__ void gtranspose(float* M, int n) {
    int bX = blockDim.x * blockIdx.x;
    int bY = blockDim.y * blockIdx.y;
    if (bY < bX) return;

    __shared__ float T[BLOCK_DIM][BLOCK_DIM+1][2+1];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int xt = blockIdx.x * blockDim.x + threadIdx.y;
    int yt = blockIdx.y * blockDim.y + threadIdx.x;


    T[threadIdx.y][threadIdx.x][0] = M[xt*n + yt];
    T[threadIdx.y][threadIdx.x][1] = M[x + y*n];
    M[x + y*n] = T[threadIdx.x][threadIdx.y][0];
    if(bY==bX)
	    return;
    __syncthreads();
    M[xt*n + yt] = T[threadIdx.y][threadIdx.x][1];
}

void transpose(float* M, int n) {
    dim3 grid(n / BLOCK_DIM, n / BLOCK_DIM, 1);
    dim3 block(BLOCK_DIM, BLOCK_DIM, 1);
    gtranspose<<<grid, block>>>(M, n);
}
#endif
