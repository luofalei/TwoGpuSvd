#include <cuda_runtime.h>
#include <math.h>
#define __CUDA_INTERNAL_COMPILATION__
#include "math_functions.h"
#undef __CUDA_INTERNAL_COMPILATION__
#include "device_launch_parameters.h"
#include <string>
#include <device_functions.h>
using namespace std;
#define TINT32 int
#define TUINT32	 unsigned int
#define TFLOAT64 double
#define TFLOAT32 float
#define FALSE false
#define TRUE true
#define PRECISION_DIVISION_LIMITATION_ZERO 1.0e-14
#define PRECISION_F 1.0e-13
#define PRECISION_RELATED_X 1.0e-14
#define PRECISION_INTERVAL 1.0e-14
#define THREADSSIZE(size) (size>512?32:16)
#define BLOCKSSIZE(size) (((size+31)/32)>128?128:((size+THREADSSIZE(size)-1)/THREADSSIZE(size)))

__device__ TINT32 _GPU_IS_INF_or_NAN(double dVal);

__device__ TFLOAT64 GPU_SecularEquation(TFLOAT64 *d, TFLOAT64 *z, TUINT32 len_d_z, TFLOAT64 x, bool &IsUnderflow);

__device__ TFLOAT64 GPU_FindOneRoot_Bisection(TFLOAT64 *d, TFLOAT64 *z, TUINT32 len_d_z, TFLOAT64 Ulimit, TFLOAT64 Dlimit,bool &IsSuccess);

__global__ void GPU_SolveSecularEquation_DO(TFLOAT64 *DEV_d, TFLOAT64 *DEV_z, TFLOAT64 *DEV_sigma, TUINT32 len_d_z
	,TFLOAT64 *DEV_ULimit, TFLOAT64 *DEV_DLimit, bool *DEV_IsSuccess
	,TUINT32 Block_Num, TUINT32 Thread_Num
	,bool UseShareMem);

__global__ void GPU_SolveSecularEquation_DO_backup(TFLOAT64 *DEV_d, TFLOAT64 *DEV_z, TFLOAT64 *DEV_sigma, TUINT32 len_d_z
	,TFLOAT64 *DEV_ULimit, TFLOAT64 *DEV_DLimit, bool *DEV_IsSuccess
	,TUINT32 Block_Num, TUINT32 Thread_Num);

bool GPU_SolveSecularEquation(TFLOAT64 *DEV_d, TFLOAT64 *DEV_z, TFLOAT64 *DEV_sigma, TUINT32 len_d_z
	,TUINT32 DevId, string &ErrorInfo);

__device__ TFLOAT64 LengthOfVector(TFLOAT64 *a, TUINT32 len);