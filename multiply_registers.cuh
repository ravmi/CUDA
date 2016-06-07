#include<stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>


#define BLOCK 32

__global__ void gmultiply(double* A, double* B, double* C, int P, int Q, int R) {
    __shared__ double At[BLOCK];
    int X = R/BLOCK;
    //int Y = P/BLOCK;
    int bY = blockIdx.x/X;
    int bX = blockIdx.x % X;
    int x0 = bX * BLOCK;
    int y0 = bY * BLOCK;
    int tix = threadIdx.x;
    int k = 0;
    double Ct[BLOCK];
#pragma unroll 32
    for(int i = 0; i < BLOCK; i++)
    {
        //          printf("wpisuje %lf do (y, x)(%d %d),  y0: %d  x0:%d, i: %d\n", Ct[i], y0+i, x0+tix, y0, x0, i);
        C[R*(y0+i) + x0 + tix] = 0;
    }

        for(int i = 0; i < BLOCK; i++)
            Ct[i] = 0.0;
    for(int G = 0; G < Q; G+=BLOCK) {
#pragma unroll 32
        for(int e=0; e < BLOCK; e++) {
            __syncthreads();
            At[tix] = A[(y0 + tix) * Q + G + e];
            __syncthreads();
            double t = B[(e + G) * R + x0 + tix];
            //    printf("watek %d %d, t:%lf, t powinno:%d, indeks:%d %d, glebokosc:%d\n", y0 + e, x0+threadIdx.x,t,R * (y0+e) + x0+threadIdx.x,y0+e+G, x0+tix, G);
#pragma unroll 32
            for(int j = 0; j < BLOCK; j++)  {
                Ct[j] += t * At[j];
                //printf("Ct[%d][%d]+=%lf * %lf, iteracja %d\n", j, tix, t, At[e][tix], e);
            }
        }
    }
#pragma unroll 32
        for(int i = 0; i < BLOCK; i++)
        {
            //printf("wpisuje %lf do (y, x)(%d %d),  y0: %d  x0:%d, i: %d\n", Ct[i], y0+i, x0+tix, y0, x0, i);
            C[R*(y0+i) + x0 + tix] += Ct[i];
        }
}

void matrixMultiply(double* A, double* B, double* C, int P, int Q, int R) {
    gmultiply<<<(P/BLOCK) *(R/BLOCK), BLOCK>>>(A, B, C, P, Q, R);
}

