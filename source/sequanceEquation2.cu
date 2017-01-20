#include "sequanceEquation2.cuh"
__device__  double add(double *z,double *d,double sigma,int size)
{
	double sum=0;
	double down;
	double up;
	double index,decimal;
	int upIndex,downIndex;
	double upDecimal,downDecimal;
	for(int i=0;i<size;++i)
	{
		down=pow(d[i],2)-pow(sigma,2);
		up=pow(z[i],2);
		if(down<MINNUM)
		{
			upDecimal=frexp(up,&upIndex);
			downDecimal=frexp(down,&downIndex);
			decimal=upDecimal/downDecimal;
			index=upIndex-downIndex;
			sum+=(decimal*pow(2.0,index));
		}
		else
		{
			sum+=(up/down);
		}
	}	
	return sum;
}

__device__ double sequanceSolve(double *z,double *d,double DLimit,double ULimit,int size)
{
	double tempu=ULimit,tempd=DLimit;
	double tempx,tempf;
	tempx=(tempu+tempd)/2.0;
	tempf=add(z,d,tempx,size);
	while(fabs(tempf)>MINNUM)
	{
		//[tempx,tempu]
		if(tempf<0)
		{
			tempd=tempx;
			tempx=(tempd+tempu)/2.0;
			if(fabs(tempx-tempu)<tempu*MINNUM)
			{
				return (tempx+tempu)/2.0;
			}
			if(fabs(tempx-tempd)<tempu*MINNUM)
			{

				return (tempx+tempd)/2.0;
			}
			tempf=add(z,d,tempx,size);
		}
		else 
		{
			tempu=tempx;
			tempx=(tempd+tempu)/2.0;
			if(fabs(tempx-tempu)<tempu*MINNUM)
			{
				return (tempx+tempu)/2.0;
			}
			if(fabs(tempx-tempd)<tempu*MINNUM)
			{
				return (tempx+tempd)/2.0;
			}
			tempf=add(z,d,tempx,size);
		}
	}
	return tempx;

}

__global__ void Solve(double *z,double *d,double *sigma,int size)
{
	int thr=threadIdx.x;
	int bl=blockIdx.x;
	int blDim=blockDim.x;
	int grDim=gridDim.x;
	for(int i=bl*blDim+thr;i<size;i+=grDim*blDim)
	{
		sigma[i]=sequanceSolve(z,d,d[i],d[i+1],size);
	}
}

void SequanceEquationSolve(double *z,double *d,double *sigma,int size)
{
	Solve<<<BLOCKSSIZE(size),THREADSSIZE(size)>>>(z,d,sigma,size);
}