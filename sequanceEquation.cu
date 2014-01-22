#include "sequanceEquation.cuh"





__device__ TINT32 _GPU_IS_INF_or_NAN(double dVal)
{
	if(__isnan(dVal) != 0)
		return 1;
	else if(__isinf(dVal) !=0)
		return 2;
	else
		return 0;
}


__device__ TFLOAT64 GPU_SecularEquation(TFLOAT64 *d, TFLOAT64 *z, TUINT32 len_d_z, TFLOAT64 x, bool &IsUnderflow)
{
	IsUnderflow = FALSE;
	TFLOAT64 res = 0;
	//TINT32  exponent1  = 0;
	//TFLOAT64 mantissa1 = 0;
	//TINT32  exponent2  = 0;
	//TFLOAT64 mantissa2 = 0;
	TINT32  exponentdown  = 0;
	TFLOAT64 mantissadown = 0;
	TINT32  exponentup  = 0;
	TFLOAT64 mantissaup = 0;
	TFLOAT64 temp = 0;
	TFLOAT64 up = 0;
	TFLOAT64 down = 0;
	for (TUINT32 i = 0; i < len_d_z; i++)
	{
		// Zk^2/(Dk^2-SIGMA^2)
		up = pow(z[i],2);
		//printf("%d\n",up);

		down = pow(d[i],2)-pow(x,2);
		//Division Precision Protection
		if(fabs(down) < PRECISION_DIVISION_LIMITATION_ZERO)    // 0.2/0.3 如果0.3太小，就用指数相乘的方式,我那个如果跳不出循环，也可以采取这种方式帮他跳出循环
		{
			mantissadown = frexp(down,&exponentdown);
			mantissaup = frexp(up,&exponentup);
			
			//*down refer to the *temp
			mantissadown = mantissaup/mantissadown;
			exponentdown = exponentup-exponentdown;

			temp = mantissadown * pow((TFLOAT64)2,exponentdown);
			res += temp;

			//IsUnderflow = TRUE;
			//return 0;
		}
		else
		{
			//printf("%d\n",down);
			temp = up/down;
			//printf("%d\n",temp);
			res += temp;
		}
		if(_GPU_IS_INF_or_NAN(res)>0)
		{
			IsUnderflow = TRUE;
			return res;
		}
	}
	res += 1;

	return res;
}


__device__ TFLOAT64 GPU_FindOneRoot_Bisection(TFLOAT64 *d, TFLOAT64 *z, TUINT32 len_d_z, TFLOAT64 Ulimit, TFLOAT64 Dlimit,bool &IsSuccess)
{
	// it need absolute precision to computing the exact root when Zi is very small
	// but double can only support related precision, so no check here, when Zi is very small
	// just return the value that close to the limitation of the interval.

	bool IsUnderflow = FALSE;


	TFLOAT64 tempu = Ulimit, tempd = Dlimit;
	TFLOAT64 tempx = (tempd + tempu) / 2.0;
	TFLOAT64 tempf = 0;
	TFLOAT64 tu,td;
	// unrecursion method
	tempf = GPU_SecularEquation(d,z,len_d_z,tempx,IsUnderflow);
	while(fabs(tempf)>PRECISION_F)
	{
		if(tempf < 0)
		{
			// find in [tempx,tempu]
			tempd = tempx;
			tempx=(tempd+tempu);
			if(fabs(tempu - tempx) < PRECISION_RELATED_X*tempu)
			{IsSuccess = TRUE;return (tempx+tempu)/2.0;}
			if(fabs(tempd - tempx) < PRECISION_RELATED_X*tempu)
			{IsSuccess = TRUE;return (tempx+tempd)/2.0;}
			tempf = GPU_SecularEquation(d,z,len_d_z,tempx,IsUnderflow);
			if (IsUnderflow)
			{
				return tempx;
			}
		}
		else
		{
			// find in [tempd,tempx]
			tempu = tempx;
			tempx = (tempu+tempd)/2.0;
			if(fabs(tempx-tempd) < PRECISION_RELATED_X*tempu)
			{IsSuccess = TRUE;return (tempd+tempx)/2.0;}
			if(fabs(tempu - tempx) < PRECISION_RELATED_X*tempu)
			{IsSuccess = TRUE;return (tempx+tempu)/2.0;}
			tempf = GPU_SecularEquation(d,z,len_d_z,tempx,IsUnderflow);
			if (IsUnderflow)
			{
				return tempx;
			}
		}
	}

	IsSuccess = TRUE;
	return tempx;
}

__device__ TFLOAT64 LengthOfVector(TFLOAT64 *a, TUINT32 len)
{
	TFLOAT64 temp = 0;
	for (TUINT32 i = 0; i < len; i++)
	{
		temp += a[i] * a[i];
	}
	return sqrt(temp);
}


__global__ void GPU_SolveSecularEquation_DO(TFLOAT64 *DEV_d, TFLOAT64 *DEV_z, TFLOAT64 *DEV_sigma, TUINT32 len_d_z
	,TFLOAT64 *DEV_ULimit, TFLOAT64 *DEV_DLimit, bool *DEV_IsSuccess
	,TUINT32 Block_Num, TUINT32 Thread_Num
	,bool UseShareMem)
{

	const int tid = threadIdx.x;
	const int bid = blockIdx.x;

	extern __shared__ TFLOAT64 share[];
	TFLOAT64 *share_d = &share[0];
	TFLOAT64 *share_z = &share_d[len_d_z];
	for (int i=bid*Thread_Num+tid; i<len_d_z;i+=Block_Num*Thread_Num)
	{
		DEV_IsSuccess[i]=false;
	}
	__syncthreads();
	if (UseShareMem)
	{
		for (int i=tid; i<len_d_z;i+=Thread_Num)
		{
			share_d[i]= DEV_d[i];
			share_z[i]= DEV_z[i];
		}
		__syncthreads();
	}


	for (int i=bid*Thread_Num+tid; i<len_d_z;i+=Block_Num*Thread_Num)
	{
		if (i<len_d_z - 1)
		{
			if (UseShareMem)
			{
				DEV_ULimit[i] = share_d[i + 1]- PRECISION_INTERVAL;
				DEV_DLimit[i] = share_d[i]+ PRECISION_INTERVAL;
			}
			else
			{
				DEV_ULimit[i] = DEV_d[i + 1]- PRECISION_INTERVAL;
				DEV_DLimit[i] = DEV_d[i]+ PRECISION_INTERVAL;
			}
			
		} 
		else
		{
			if (UseShareMem)
			{
				DEV_ULimit[i] = share_d[i] + LengthOfVector(share_z,len_d_z) - PRECISION_INTERVAL;
				DEV_DLimit[i] = share_d[i]+ PRECISION_INTERVAL;
			}
			else
			{
				DEV_ULimit[i] = DEV_d[i] + LengthOfVector(DEV_z,len_d_z) - PRECISION_INTERVAL;
				DEV_DLimit[i] = DEV_d[i]+ PRECISION_INTERVAL;
			}
		}
	}

	__syncthreads();

	for (int i=bid*Thread_Num+tid; i<len_d_z;i+=Block_Num*Thread_Num)
	{
		if (UseShareMem)
			DEV_sigma[i] = GPU_FindOneRoot_Bisection(share_d,share_z,len_d_z,DEV_ULimit[i],DEV_DLimit[i],DEV_IsSuccess[i]);
		else
			DEV_sigma[i] = GPU_FindOneRoot_Bisection(DEV_d,DEV_z,len_d_z,DEV_ULimit[i],DEV_DLimit[i],DEV_IsSuccess[i]);
	}
	__syncthreads();

}


__global__ void GPU_SolveSecularEquation_DO_backup(TFLOAT64 *DEV_d, TFLOAT64 *DEV_z, TFLOAT64 *DEV_sigma, TUINT32 len_d_z
	,TFLOAT64 *DEV_ULimit, TFLOAT64 *DEV_DLimit, bool *DEV_IsSuccess
	,TUINT32 Block_Num, TUINT32 Thread_Num)
{
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;


	for (int i=bid*Thread_Num+tid; i<len_d_z;i+=Block_Num*Thread_Num)
	{
		if (i<len_d_z - 1)
		{

			DEV_ULimit[i] = DEV_d[i + 1]- PRECISION_INTERVAL;
			DEV_DLimit[i] = DEV_d[i]+ PRECISION_INTERVAL;
		} 
		else
		{

			DEV_ULimit[i] = DEV_d[i] + LengthOfVector(DEV_z,len_d_z) - PRECISION_INTERVAL;
			DEV_DLimit[i] = DEV_d[i]+ PRECISION_INTERVAL;
		}
	}

	__syncthreads();

	for (int i=bid*Thread_Num+tid; i<len_d_z;i+=Block_Num*Thread_Num)
	{
		DEV_sigma[i] = GPU_FindOneRoot_Bisection(DEV_d,DEV_z,len_d_z,DEV_ULimit[i],DEV_DLimit[i],DEV_IsSuccess[i]);
	}
	__syncthreads();

}


bool GPU_SolveSecularEquation(TFLOAT64 *DEV_d, TFLOAT64 *DEV_z, TFLOAT64 *DEV_sigma, TUINT32 len_d_z
	,TUINT32 DevId, string &ErrorInfo)
{

	TUINT32 time_start;
	TUINT32 time_end;
	TUINT32 Block_Num,Thread_Num;
	TUINT32 ShareMemSize;
	cudaError_t cudaStat;
//	cudaStat = cudaSetDevice(DevId);

	// debug
	TFLOAT64 *debug = new TFLOAT64[len_d_z];

	// DEV_IsSuccess,DEV_ULimit,DEV_DLimit are temporary variables need to be free when finish
	// this call
	bool	 *DEV_IsSuccess, *CUP_IsSuccess;
	TFLOAT64 *DEV_ULimit, *DEV_DLimit;
	cudaStat = cudaMalloc((void**)&DEV_IsSuccess,sizeof(bool)*len_d_z);
	if (cudaStat != cudaSuccess)
	{ ErrorInfo = "Device malloc Error.\n"; return FALSE;}
	cudaStat = cudaMalloc((void**)&DEV_ULimit,sizeof(TFLOAT64)*len_d_z);
	if (cudaStat != cudaSuccess)
	{ ErrorInfo = "Device malloc Error.\n"; return FALSE;}
	cudaStat = cudaMalloc((void**)&DEV_DLimit,sizeof(TFLOAT64)*len_d_z);
	if (cudaStat != cudaSuccess)
	{ ErrorInfo = "Device malloc Error.\n"; return FALSE;}


	Thread_Num=THREADSSIZE(len_d_z);
	Block_Num=THREADSSIZE(len_d_z);
	//ShareMemSize = 0;
	ShareMemSize = sizeof(TFLOAT64)*len_d_z*2;
	//Block_Num=2;Thread_Num=3;
	//bool UseShareMem = ShareMemSize < 16000;
	bool UseShareMem=0;
	GPU_SolveSecularEquation_DO<<<Block_Num,Thread_Num,UseShareMem?ShareMemSize:0>>>(DEV_d,DEV_z,DEV_sigma,len_d_z,
		DEV_ULimit,DEV_DLimit,DEV_IsSuccess
		,Block_Num,Thread_Num
		,UseShareMem
		);

	CUP_IsSuccess = new bool[len_d_z];

	cudaStat = cudaMemcpy(CUP_IsSuccess,DEV_IsSuccess,sizeof(bool)*len_d_z,cudaMemcpyDeviceToHost);

	// chech the result
	for (int i=0;i<len_d_z;i++)
	{
		if (!CUP_IsSuccess[i])
		{
			ErrorInfo = "Error occured in the device when solve equation.\n";
			return FALSE;
		}
	}


	// Release Resource
	cudaFree(DEV_IsSuccess);
	cudaFree(DEV_ULimit);
	cudaFree(DEV_DLimit);
	delete [] CUP_IsSuccess;

	
	return TRUE;
}
