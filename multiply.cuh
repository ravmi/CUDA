#include <cuda.h>
#include <cuda_runtime.h>


#define BLOCK 32

__global__ void gmultiply(double* A, double* B, double* C, int P, int Q, int R) {
    __shared__ double4 At[BLOCK/2][BLOCK/2];
    __shared__ double4 Bt[BLOCK/2][BLOCK/2];
    int x0 = blockIdx.x * BLOCK;
    int y0 = blockIdx.y * BLOCK;
    int tix = threadIdx.x;
    int tiy = threadIdx.y;
    int tix1 = tix*2;
    int tix2 = tix*2 + 1;
    int tiy1 = tiy*2;
    int tiy2 = tiy*2 + 1;
    int x1 = (blockIdx.x*BLOCK/2 + threadIdx.x) * 2;
    int x2 = (blockIdx.x*BLOCK/2 + threadIdx.x) * 2 + 1;
    int y1 = (blockIdx.y*BLOCK/2 + threadIdx.y) * 2;
    int y2 = (blockIdx.y*BLOCK/2 + threadIdx.y) * 2 + 1;
    if(x2 < R && y2 < P) {


/*
    double c[CUT];
    for(int i = 0 ; i < CUT; i++)
    */
        double c1 = 0;
        double c2 = 0;
        double c3 = 0;
        double c4 = 0;

    for(int G = 0; G < Q; G+=BLOCK) {
        __syncthreads();
        At[tiy][tix].x = A[(y1) * Q + tix1 + G];
        At[tiy][tix].y = A[(y1) * Q + tix2 + G];
        At[tiy][tix].z = A[(y2) * Q + tix1 + G];
        At[tiy][tix].w = A[(y2) * Q + tix2 + G];

        Bt[tiy][tix].x = B[(tiy1 + G) * R + x1];
        Bt[tiy][tix].y = B[(tiy1 + G) * R + x2];
        Bt[tiy][tix].z = B[(tiy2 + G) * R + x1];
        Bt[tiy][tix].w = B[(tiy2 + G) * R + x2];
        //Bt[tiy][tix2].w = B[(tiy + G) * R + tix + x0];
        //printf("wpisano do A[%d][%d]: %lf,    do B[%d][%d]: %lf, block id:: (%d, %d) iteracja:%d\n", tiy , tix ,At[tiy][tix] , tiy , tix, Bt[tiy][tix],blockIdx.y,     blockIdx.x, G);
        __syncthreads();
        //    printf("watek %d %d, t:%lf, t powinno:%d, indeks:%d %d, glebokosc:%d\n", y0 + e, x0+threadIdx.x,t,R * (y0+e) + x0+threadIdx.x,y0e+G, x0+tix, G);
#pragma unroll 16
        for(int i = 0; i < BLOCK/2; i++)  {
            //__syncthreads();
            c1+=At[tiy][i].x * Bt[i][tix].x + At[tiy][i].y * Bt[i][tix].z;
            c2+=At[tiy][i].x * Bt[i][tix].y + At[tiy][i].y * Bt[i][tix].w;
            c3+=At[tiy][i].z * Bt[i][tix].x + At[tiy][i].w * Bt[i][tix].z;
            c4+=At[tiy][i].z * Bt[i][tix].y + At[tiy][i].w * Bt[i][tix].w;
//          printf("Mnozymy: At[%d][%d] * Bt[%d][%d] ++= %lf, powinno byc %lf\n", tiy,i,i,tix, c, A[tiy * Q + tix + G + i] *B[(tiy + G + i) * R + tix]);
            //printf("Ct[%d][%d]+=%lf * %lf, iteracja %d\n", j, tix, t, At[e][tix], e);
        }
        //__syncthreads();
        //__syncthreads();
    }
    C[R*(y1) + x1] = c1;
    C[R*(y1) + x2] = c2;
    C[R*(y2) + x1] = c3;
    C[R*(y2) + x2] = c4;
    }
}

void matrixMultiply(double* A, double* B, double* C, int P, int Q, int R) {
    dim3 block(BLOCK/2, BLOCK/2, 1);
    dim3 grid((R/2 + BLOCK/2 - 1)/(BLOCK/2), (P/2 + BLOCK/2  - 1)/(BLOCK/2));
    gmultiply<<<grid, block>>>(A, B, C, P, Q, R);
}

