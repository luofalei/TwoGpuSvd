#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cublas.h"
#include "cublas_v2.h"
#include <stdlib.h>
#include <time.h>
#define IDX2C(i,j,leading) ((i)+((j)*(leading)))
typedef struct _data
{
	int rows;
	int cols;
	double* data;
} Metrix;
typedef struct _data * PMetrix;
void metrixMultiply(PMetrix a,PMetrix b ,PMetrix c);