#include<stdio.h>
#include<stdlib.h>

#define CUDA_CHECK_RETURN(value) { \
cudaError_t _m_cudaStat = value; \
if (_m_cudaStat != cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
        exit(1); \
} }

#define BLOCK_SIZE_LIMIT (1024)
#define GRID_SIZE_LIMIT (65535)

__global__ void multiply(int N, int dsize, int *M, int *D, int* T, int *R) {
    for (int rindex = blockIdx.x * blockDim.x + threadIdx.x; rindex < N;
	    rindex += gridDim.x * blockDim.x) {
	for(int i = 0; i < dsize; i++) {
	    int tindex = D[i] + rindex;
	    if(tindex>=0 && tindex < N)
		R[rindex] += *(M + rindex * dsize + i) * T[tindex];
	}
    }
}

void clean(int *T, int n) {
    for(int i = 0; i < n; i++)
	T[i] = 0;
}

void print_array(int *T, int n) {
    for(int i = 0; i < n; i++)
	printf("%d\n", T[i]);
}

int main() {
    int n;
    scanf("%d", &n);

    // length of the vector describing input matrix
    int dsize;
    scanf("%d", &dsize);

    if(n==0)
	return 0;

    int block_size = BLOCK_SIZE_LIMIT;
    int threads = min(n, GRID_SIZE_LIMIT * block_size);
    int grid_size = (threads+block_size-1)/block_size;

    // Matrix
    int *mat;
    int *devMat;
    cudaMallocHost((void**) &mat, n * dsize * sizeof(int));
    cudaMalloc((void**) &devMat, n * dsize * sizeof(int));

    // Vector describing matrix
    int *D;
    int *devD;
    cudaMallocHost((void**) &D, dsize * sizeof(int));
    cudaMalloc((void**) &devD, dsize * sizeof(int));

    // Vector we multiply by (Times)
    int *T;
    int *devT;
    cudaMallocHost((void**) &T, n * sizeof(int));
    cudaMalloc((void**) &devT, n * sizeof(int));

    // result
    int *R;
    int *devR;
    cudaMallocHost((void**) &R, n * sizeof(int));
    cudaMalloc((void**) &devR, n * sizeof(int));
    clean(R, n);

    for(int i = 0; i < dsize; i++) {
	scanf("%d", &D[i]);
	for(int j = 0; j < n; j++)
	    scanf("%d", mat+j*dsize + i);
	// n rows, dsize columns
	// each column is some diagonal of original matrix
    }

    for(int i = 0; i < n; i++)
	scanf("%d", &T[i]);

    cudaMemcpy(devMat, mat, sizeof(int) * n *dsize, cudaMemcpyHostToDevice);
    cudaMemcpy(devD, D, sizeof(int) * dsize, cudaMemcpyHostToDevice);
    cudaMemcpy(devT, T, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(devR, R, sizeof(int) * n, cudaMemcpyHostToDevice);

    multiply<<<grid_size, block_size>>>(n, dsize, devMat, devD, devT, devR);

    CUDA_CHECK_RETURN(cudaGetLastError());

    cudaMemcpy(R, devR, sizeof(int) * n, cudaMemcpyDeviceToHost);
    print_array(R, n);

    cudaFreeHost((void*) mat);
    cudaFree((void*) devMat);
    cudaFreeHost((void*) D);
    cudaFree((void*) devD);
    cudaFreeHost((void*) T);
    cudaFree((void*) devT);
    cudaFreeHost((void*) R);
    cudaFree((void*) devR);

    return 0;
}
