#include <stdio.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef THREADSSIZE(size)
#define THREADSSIZE(size) (size>512?32:16)
#endif
#ifndef BLOCKSSIZE(size)
#define BLOCKSSIZE(size) (((size+31)/32)>128?128:((size+THREADSSIZE(size)-1)/THREADSSIZE(size)))
#endif
#ifndef MINNUM 
#define MINNUM 1.0e-14
#endif
__global__ void Solve(double *z,double *d,double *sigma,int size);
__device__ double sequanceSolve(double *z,double *d,double DLimit,double ULimit,int size);
__device__  double add(double *z,double *d,double sigma,int size);
void SequanceEquationSolve(double *z,double *d,double *sigma,int size);