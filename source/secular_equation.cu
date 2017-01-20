#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include "secular_equation.cuh"
#include "devide_metrix.h"
#include <math_functions.h>
#include <helper_cuda.h>
#include <string.h>
#include "print_metrix.h"
#include <device_functions.h>
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#define MAXSHARE 2000
#define MINACCURACY 1.0e-14
#define ACCURACYMIN 1.0e-14
#define PRECISION_RELATED_X 1.0e-14
#define PRECISION_INTERVAL 1.0e-14
#define BLOCK_SIZE 16
#define PRECISION_DIVISION_LIMITATION_ZERO 1.0e-14
#define PRECISION_COMPARE_LIMITATION_ZERO 1.0e-2
#define BLOCKMIN(size)  (((size+255)/256)>192?192:((size+255)/256))
#define THREADMIN(size) (size>256?256:size)
#define MAXSM 1000



bool   produce_mid_metrix(GPUSave *temp,int DeviceID,bool result_type)
{
	double *temp_z;
	double *temp_d;
	double *mid_z;
	double *mid_d;
	double *temp_w;
	double *result_Q;
	double *mid_Q;
	double *mid_q;
	int size;
	double temp_r;
	double temp_c0;
	double temp_s0;
	double *mid_W;
	double *result_W;
	cudaSetDevice(DeviceID);
	size = temp->left_size+temp->right_size+1;
	temp->size = size;
//	FILE *ph=fopen("F:cheack.txt","w");
//	print_mid_metrix(ph,temp->left_Q,temp->left_q,temp->left_D,temp->left_W,temp->left_size);
//	print_mid_metrix(ph,temp->right_Q,temp->right_q,temp->right_D,temp->right_W,temp->right_size);
//	fclose(ph);

	temp_z = (double *)malloc(sizeof(double)*size);
	temp_d = (double*)malloc(sizeof(double)*(size+1));
	mid_z = (double *)malloc(sizeof(double )*size);
	mid_d=(double*)malloc(sizeof(double)*(size+1));
	temp_w=(double*)malloc(sizeof(double)*size);
	
	/*求解中间矩阵*/
	bool rights = true;
	rights=formM(temp->ak,temp->bk,temp->left_Q,temp->left_q,\
		temp->left_D,temp->right_Q,temp->right_q,temp->right_D,\
		temp_z,temp_d,temp->left_size,temp->right_size,temp_r);


	

	free(temp->left_D);
	free(temp->right_D);

	if(rights ==false)
	{
		printf("Error: in secular_equation 58");
		return false;
	}
	/*求c0,s0*/
	if (temp_r<0)
	{
		printf("Error: the error in the function of produce_mid_metrix of the secular_equation\n ");
		return false;
	}
	if(temp_r<MINACCURACY)
	{
		temp_r= MINACCURACY;
	}
	temp_c0 = temp->ak*temp->left_q[temp->left_size]/temp_r;
	temp_s0 = temp->bk*temp->right_q[0]/temp_r;

	/*给d进行排序*/
	sortM(size,temp_z,temp_d,mid_z,mid_d);

	//debug
//	FILE *phf=fopen("F:zandd.txt","w");
//	fprintf(phf,"%d\n",size);
//	for(int i=0;i<size;++i)
//	{
//		fprintf(phf,"%lf %lf\n",mid_z[i],mid_d[i]);
//	}
//	fclose(phf);
	//debug



	/*求w*/
	mid_d[size]=mid_d[size-1]+geometric_z(mid_z,size);
	if(!mid_function(mid_d,mid_z,temp_w,size))
	{
		printf("error in sequance equation solve!\n");
		return false;
	}
	/*free*/
	free(mid_z);
	free(mid_d);
	/*构造Q W*/
	mid_Q = (double*)malloc(sizeof(double)*(size+1)*size);
	produce_Q(temp,mid_Q,temp_c0,temp_s0);
	free(temp->left_Q);
	free(temp->right_Q);


	mid_W = (double*)malloc(sizeof(double)*size*size);
	produce_W(temp,mid_W);
	free(temp->left_W);
	free(temp->right_W);

	mid_q = (double*)malloc(sizeof(double)*(size+1));
	produce_q(temp,mid_q,temp_c0,temp_s0);
	free(temp->left_q);
	free(temp->right_q);

	/*校正z，求u，v，同时让u*Q   v*W */
	result_Q=(double*)malloc(sizeof(double)*size*(size+1));
	result_W=(double*)malloc(sizeof(double)*size*size);

	mid_function_two(temp_d,temp_z,temp_w,mid_Q,mid_W,result_Q,result_W,size);
	if(!result_type)
	{
		GPUSave *fa_temp;
		fa_temp=temp->father;
		if(temp->leaf_left==true)
		{
			fa_temp->left_Q=result_Q;
			fa_temp->left_W=result_W;
			fa_temp->left_D=temp_w;
			fa_temp->left_q=mid_q;
			fa_temp->left_size=size;
		}
		else
		{
			fa_temp->right_Q=result_Q;
			fa_temp->right_W=result_W;
			fa_temp->right_D=temp_w;
			fa_temp->right_q=mid_q;
			fa_temp->right_size=size;
		}
		free(temp_z);
		free(temp_d);
		free(mid_W);
		free(mid_Q);
		delete temp;
	}
	else
	{
		temp->self_Q=result_Q;
		temp->self_W=result_W;
		temp->self_q=mid_q;
		temp->self_D=temp_w;
		temp->size=size;
	
		free(temp_z);
		free(temp_d);
		free(mid_W);
		free(mid_Q);
	}

		/*free*/

		return true;
	
}



double geometric_z(double *mid_z,int size)
{
	double all=0;
	for(int i=0;i<size;i++)
	{
		all+=pow(mid_z[i],2);
	}
	return sqrt(all);
}
		














/****************************************************************************************************************************************************
**************************************中间函数，主函数通过中间函数沟通gpu，这个包括申请内存啥的，为的是让*********************************
**************************************主函数个子小一点，如果有问题它上面有很大原因************************************************************
****************************************************************************************************************************************************/



bool mid_function_two(double *temp_d,double * temp_z,double *temp_w,double *Q,double *W,\
	double *result_Q,double *result_W,int size)
{
	double *dev_d;
	double *dev_z;
	double *result_z;
	double *dev_u;
	double *dev_v;
	double *dev_w;
	double *tz=(double*)malloc(sizeof(double)*size);
	checkCudaErrors(cudaMalloc((void**)&dev_d,sizeof(double)*size));
	checkCudaErrors(cudaMalloc((void**)&dev_z,sizeof(double)*size));
	checkCudaErrors(cudaMalloc((void**)&result_z,sizeof(double)*size));
	checkCudaErrors(cudaMalloc((void**)&dev_w,sizeof(double)*size));
	
	checkCudaErrors(cudaMemcpy(dev_d,temp_d,sizeof(double)*size,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_z,temp_z,sizeof(double)*size,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_w,temp_w,sizeof(double)*size,cudaMemcpyHostToDevice));
	//UpdateZ<<<16,16>>>(dev_d,dev_w,dev_z,result_z,size);
	adjust_z<<<BLOCKMIN(size),THREADMIN(size)>>>(dev_d,dev_w,dev_z,result_z,size);
	//test
	//checkCudaErrors(cudaMemcpy(tz,result_z,sizeof(double)*size,cudaMemcpyDeviceToHost));
	//test
	//checkCudaErrors(cudaFree(dev_z));

	checkCudaErrors(cudaMalloc((void**)&dev_u,sizeof(double)*size*size));
	checkCudaErrors(cudaMalloc((void**)&dev_v,sizeof(double)*size*size));
	//我现在是传的没有校正过的z
	get_u_v<<<BLOCKMIN(size),THREADMIN(size)>>>(dev_w,dev_z,dev_d,dev_u,dev_v,size);
//	double *tempGetV=(double *)malloc(sizeof(double)*size*size);
//	checkCudaErrors(cudaMemcpy(tempGetV,dev_v,sizeof(double)*size*size,cudaMemcpyDeviceToHost));
//	checkCudaErrors(cudaFree(dev_v));
//	double *tempGetU=(double *)malloc(sizeof(double)*size*size);
//	checkCudaErrors(cudaMemcpy(tempGetU,dev_u,sizeof(double)*size*size,cudaMemcpyDeviceToHost));
//	checkCudaErrors(cudaFree(dev_u));
//	Q=encode(Q,size,size+1);
//	tempGetU=encode(tempGetU,size,size);
//	tempGetV=encode(tempGetV,size,size);
//	W=encode(W,size,size);
//	*result_Q=mulMetrixSove(Q,tempGetU,size,size+1,size);
//	
//	free(tempGetU);
//	free(Q);
//	*result_W=mulMetrixSove(W,tempGetV,size,size,size);
//	free(tempGetV);
//	free(W);
	//double *cheackU = (double *)malloc(sizeof(double)*size*size);
	//test
//	cudaMemcpy(cheackU,dev_u,sizeof(double)*size*size,cudaMemcpyDeviceToHost);
//	cudaMemcpy(cheackU,dev_v,sizeof(double)*size*size,cudaMemcpyDeviceToHost);
	//test
//	FILE *ph=fopen("F:u.txt","w");
//	print_block(ph,cheackU,size,size);
//	fclose(ph);

	double *d_Q,*d_reQ;
	checkCudaErrors(cudaMalloc((void**)&d_Q,sizeof(double)*size*(size+1)));
	checkCudaErrors(cudaMalloc((void**)&d_reQ,sizeof(double)*size*(size+1)));
	checkCudaErrors(cudaMemcpy(d_Q,Q,sizeof(double)*size*(size+1),cudaMemcpyHostToDevice));
//	int grid_y=((1+size)%BLOCK_SIZE)?(1+size+BLOCK_SIZE):(1+size);
//	int grid_x=(size%BLOCK_SIZE)?(BLOCK_SIZE+size):size;
	int grid_y=((size+BLOCK_SIZE)/BLOCK_SIZE);
	int grid_x=((size+BLOCK_SIZE-1)/BLOCK_SIZE);
	dim3 griddim(grid_x,grid_y);
	dim3 blockdim(BLOCK_SIZE,BLOCK_SIZE);

	//释放空间V
	double *tempGetV=(double *)malloc(sizeof(double)*size*size);
	checkCudaErrors(cudaMemcpy(tempGetV,dev_v,sizeof(double)*size*size,cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(dev_v));
	//释放完毕

	d_metrix_multi<<<griddim,blockdim>>>(d_Q,dev_u,d_reQ,size,size,size+1);
	checkCudaErrors(cudaMemcpy(result_Q,d_reQ,sizeof(double)*size*(size+1),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_Q));
	checkCudaErrors(cudaFree(d_reQ));
	checkCudaErrors(cudaFree(dev_u));

	double *d_W,*d_reW;
	checkCudaErrors(cudaMalloc((void**)&d_W,sizeof(double)*size*size));
	checkCudaErrors(cudaMalloc((void**)&d_reW,sizeof(double)*size*size));
	checkCudaErrors(cudaMemcpy(d_W,W,sizeof(double)*size*size,cudaMemcpyHostToDevice));
	
	grid_y=((size+BLOCK_SIZE-1)/BLOCK_SIZE);
	grid_x=((size+BLOCK_SIZE-1)/BLOCK_SIZE);
	dim3 griddimw(grid_x,grid_y);
	dim3 blockdimw(BLOCK_SIZE,BLOCK_SIZE);

	checkCudaErrors(cudaMalloc((void**)&dev_v,sizeof(double)*size*size));
	checkCudaErrors(cudaMemcpy(dev_v,tempGetV,sizeof(double)*size*size,cudaMemcpyHostToDevice));
	free(tempGetV);
	d_metrix_multi<<<griddimw,blockdimw>>>(d_W,dev_v,d_reW,size,size,size);
	checkCudaErrors(cudaMemcpy(result_W,d_reW,sizeof(double)*size*size,cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_W));
	checkCudaErrors(cudaFree(d_reW));
	checkCudaErrors(cudaFree(dev_v));

	
	return true;




}
double * encode(double *Q,int cols,int rows)
{
	double *temp=(double *)malloc(sizeof(double)*cols*rows);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			temp[IDX2C(i,j,rows)]=Q[i*cols+j];
		}
	}
	free(Q);
	return temp;
}
double * decode(double *Q,int cols,int rows)
{
	double *temp=(double *)malloc(sizeof(double)*cols*rows);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			temp[i*cols+j]=Q[IDX2C(i,j,rows)];
		}
	}
	free(Q);
	return temp;
}


double * mulMetrixSove(double *Q,double *W,int lcol,int lrow,int rcol)
{

	PMetrix metrixQ=(PMetrix)malloc(sizeof(Metrix));
	PMetrix metrixU=(PMetrix)malloc(sizeof(Metrix));
	PMetrix resultMeQ=(PMetrix)malloc(sizeof(Metrix));
	metrixQ->data=Q;
	metrixQ->cols=lcol;
	metrixQ->rows=lrow;
	metrixU->data=W;
	metrixU->rows=lcol;
	metrixU->cols=rcol;
	resultMeQ->cols=metrixU->cols;
	resultMeQ->rows=metrixQ->rows;
	double *result=(double*)malloc(sizeof(double)*resultMeQ->cols*resultMeQ->rows);
	resultMeQ->data=result;
	metrixMultiply(metrixQ,metrixU,resultMeQ);
	return decode(resultMeQ->data,resultMeQ->cols,resultMeQ->rows);
}





bool mid_function(double *mid_d,double *mid_z,double *w,int size)
{
	double * dev_d;
	double * dev_z;
	double * dev_w;
	string error;
	bool RIGHT;
	checkCudaErrors(cudaMalloc((void**)&dev_d,sizeof(double)*(size+1)));
	checkCudaErrors(cudaMalloc((void **)&dev_z,sizeof(double)*size));
	checkCudaErrors(cudaMalloc((void**)&dev_w,sizeof(double)*size));
	checkCudaErrors(cudaMemcpy(dev_d,mid_d,(size+1)*sizeof(double),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_z,mid_z,sizeof(double)*size,cudaMemcpyHostToDevice));
	SequanceEquationSolve(dev_z,dev_d,dev_w,size);
	//RIGHT=GPU_SolveSecularEquation(dev_d,dev_z,dev_w,size,0,error);
	//devide_gpu_thread(MAXNUM,threads,block);
//	double *temps =(double*)malloc(sizeof(double)*size);
//	double *temps_dev;
//	checkCudaErrors(cudaMalloc((void**)&temps_dev,size*sizeof(double)));

	//secular_equation_small<<<32,128>>>(dev_z,dev_d,dev_w,size,temps_dev,memZ,memD);
	checkCudaErrors(cudaMemcpy(w,dev_w,size*sizeof(double),cudaMemcpyDeviceToHost));
	double tempW;
	for(int i=0;i<size/2;++i)
	{
		tempW=w[i];
		w[i]=w[size-1-i];
		w[size-1-i]=tempW;
	}
	cudaFree(dev_d);
	cudaFree(dev_w);
	cudaFree(dev_z);
	return true;

}






bool  formM(double alpha,double beta,double *Q1,double *q1,\
	double *D1,double *Q2,double *q2,double *D2,double *z,double *d,\
	int up_number,int down_number,double &gama)

{/*The following is formM function.Input alpha,beta,and the first 2 matrix of B1's svd,and the first 2 matrix of B2's svd,and n*/
	/*so we can get the middle matrix M (also z[n] and d[n], in which d[0]=0)*/  

	double last1,first2;
	/*The fowlloing is form the releted matrix */
	
	if(q1==NULL||Q1==NULL||D1==NULL||Q2==NULL||q2==NULL||D2==NULL||z==NULL||d==NULL)
	{
		printf("Error: in the secular_equation of formM\n");
		return false;
	}

	last1=q1[up_number];   first2=q2[0]; /*last1,first2*/

	gama=sqrt((alpha*last1)*(alpha*last1)+(beta*first2)*(beta*first2));  /*gama*/

	/*Form matrix M (z[i] and d[i])*/
	z[0]=gama;
	for(int i=0;i<up_number;i++)
	{
		z[i+1]=alpha*Q1[(up_number)*(up_number)+i];
	}
	for(int i=0;i<down_number;i++)
	{
		z[i+up_number+1]=beta*Q2[i];   /*z[i] is the first columns of matrix M*/
	}
	d[0]=0;
	for(int i=0;i<up_number;i++)
	{	
		d[i+1]=D1[i];
	}/* the No.(i-1) singler of B1*/
	for(int i=0;i<down_number;i++)
	{
		d[i+up_number+1]=D2[i];
	}/* the No.(i-k) singler of B2*/
	return true;
}

/*sort z[n] and d[n] of Matrix M, and d[n] adds another element for the secular equation*/
bool  sortM(int n,double *z,double *d,double *znew,double*dnew)  
{
	for(int i=0;i<n;++i)
	{
		znew[i]=z[i];
		dnew[i]=d[i];
	}
	mergeSort(dnew,n,znew);
	return 1;
}
void mergeSort(double *Data,int size,double *z)
{
	if(size==1)
	{
		return ;
	}
	double *left,*right,*Zleft,*Zright;
	int leftSize=size/2;
	int rightSize=size-size/2;
	left=Data;
	right=Data+leftSize;
	Zleft=z;
	Zright=z+leftSize;
	mergeSort(left,leftSize,Zleft);
	mergeSort(right,rightSize,Zright);
	merge(left,leftSize,right,rightSize,Zleft,Zright);
	
}
void merge(double *left,int leftSize,double *right,int rightSize,double *Zleft,double *Zright)
{
	double *result=(double *)malloc(sizeof(double)*(leftSize+rightSize));
	double *resultPos=(double *)malloc(sizeof(double)*(leftSize+rightSize));
	int leTop=0,riTop=0,top=0;
	while(leTop<leftSize&&riTop<rightSize)
	{
		if(left[leTop]<right[riTop])
		{
			resultPos[top]=Zleft[leTop];
			result[top++]=left[leTop++];
			
		}
		else
		{
			resultPos[top]=Zright[riTop];
			result[top++]=right[riTop++];
		}
	}
	while(leTop<leftSize)
	{
		resultPos[top]=Zleft[leTop];
		result[top++]=left[leTop++];
	}
	while(riTop<rightSize)
	{
		resultPos[top]=Zright[riTop];
		result[top++]=right[riTop++];
	}
	for(int i=0;i<leftSize;++i)
	{
		left[i]=result[i];
		Zleft[i]=resultPos[i];
	}
	for(int i=0;i<rightSize;++i)
	{
		right[i]=result[leftSize+i];
		Zright[i]=resultPos[leftSize+i];
	}
	free(result);
	free(resultPos);
}
__global__ void secular_equation_small(double *z,double *d,double *w,int size,double *temps,double *memZ,double *memD)
{
//	extern __shared__ double sh_temp[];
//	double * temp_d = sh_temp;
//	double * temp_z = (double*)&temp_d[size+1];
	int thread_num;
	int block_num;
	thread_num=threadIdx.x;
	block_num = blockIdx.x;
	int IsSuccess;
	double mid_w;
	for(int i=thread_num ; i<size ; i+=blockDim.x)
	{
		memD[i]=d[i]*d[i];
		memZ[i]=z[i]*z[i];
	}
	if(thread_num==0)
	{
		d[size]=d[size]*d[size];
	}
	__syncthreads();
	for(int i=thread_num+block_num*blockDim.x;i<size; i+= blockDim.x * gridDim.x)
	{
		
		mid_w=GPU_FindOneRoot_Bisection(memD,memZ,size,d[i+1],d[i],IsSuccess,temps[i]);
		if(IsSuccess==true)
			w[i]=mid_w;
		else
			w[i]=10;
		
	}
}

/*给的都是平方  d2,z2,w2，节约计算*/

//__global__ void seqularBigData(double *z,double *d,double *w,int size,double *temps)
//{
//	extern __shared__ double SMtemp[];
//	double *tempD = SMtemp;
//	double *tempZ = &SMtemp[MAXSM];
//	int threadPos = threadIdx.x;
//	int blockPos  = blockIdx.x;
//	
//	for(int i=0;i<MAXSM;i+= blockDim.x)
//	{
//		tempD[i] = d[i]*d[i];
//		tempZ[i] = z[i]*z[i];
//	}
//	for(int i=blockPos*blockDim.x+threadIdx.x;i<=size;i+=gridDim.x*blockDim.x)
//	{
//		if(i>MAXSM)
//		{
//			d[i]=d[i]*d[i];
//			z[i]=z[i]*z[i];
//		}
//	}
//	__syncthreads();
//}


/*
__device__ double add_secular(double *temp_d,double *temp_z,int size,double w)
{
	double res = 0;
	//TINT32  exponent1  = 0;
	//TFLOAT64 mantissa1 = 0;
	//TINT32  exponent2  = 0;
	//TFLOAT64 mantissa2 = 0;
	int  exponentdown  = 0;
	double mantissadown = 0;
	int  exponentup  = 0;
	double mantissaup = 0;
	double temp = 0;
	double up = 0;
	double down = 0;
	for (int i = 0; i < size; i++)
	{
		// Zk^2/(Dk^2-SIGMA^2)
		up = temp_z[i];
		//printf("%d\n",up);

		down = temp_d[i]-w;
		//Division Precision Protection
		if(abs(down) < PRECISION_DIVISION_LIMITATION_ZERO)
		{
			mantissadown = frexp(down,&exponentdown);
			mantissaup = frexp(up,&exponentup);
			
			//*down refer to the *temp
			mantissadown = mantissaup/mantissadown;
			exponentdown = exponentup-exponentdown;

			temp = mantissadown * pow(2.0,exponentdown);
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
	}
	res += 1;

	return res;
}
*/





__device__  double  add_secular(double *temp_d,double *temp_z,int size,double w)
{
	double __equal=0;
	double __down;
	double __square_z;
	if(size<0)
	{
		return false;
	}
	for(int i=0;i<size;++i)
	{
		__down = temp_d[i]-w;
		if(__down<MINACCURACY&&__down>=0)
		{
			__down=MINACCURACY;
		}
		else if(__down>-MINACCURACY&&__down<=0)
		{
			__down=-MINACCURACY;
		}
		__square_z=temp_z[i];
		__equal+= __square_z/__down;
	}
	return  __equal+1.0;
}



__device__ double  GPU_FindOneRoot_Bisection(double  *d, double  *z, int  len_d_z, double  Ulimit, double  Dlimit,int  &IsSuccess,double &tempfs)
{
	// it need absolute precision to computing the exact root when Zi is very small
	// but double can only support related precision, so no check here, when Zi is very small
	// just return the value that close to the limitation of the interval.



	double  tempu = Ulimit, tempd = Dlimit;   //change right
	double  tempx = (tempd + tempu) / 2.0;
	double  tempf = 0;

	// unrecursion method
	tempf = add_secular(d,z,len_d_z,tempx*tempx);
	while(fabs(tempf)>PRECISION_F)
	{
		if(tempf < 0)
		{
			// find in [tempx,tempu]
			tempd = tempx;
			tempx = (tempu+tempd)/2.0;
			if(tempx==tempu||tempx==tempd)
			{
				return tempx;
			}
			if(fabs(tempu - tempx) < PRECISION_RELATED_X*tempu)
			{IsSuccess = TRUE;return (tempx+tempu)/2.0;}
			tempf = add_secular(d,z,len_d_z,tempx*tempx); //change here
		}
		else
		{
			// find in [tempd,tempx]
			tempu = tempx;
			tempx = (tempu+tempd)/2.0;
			if(tempx==tempu||tempx==tempd)
			{
				return tempx;
			}
			if(fabs(tempx-tempd) < PRECISION_RELATED_X*tempu)
			{IsSuccess = TRUE;return (tempd+tempx)/2.0;}
			tempf = add_secular(d,z,len_d_z,tempx*tempx);
		}
	}
	
	IsSuccess = TRUE;
	tempfs = tempf;
	return tempx;
}






/*/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////*/
/*下面是求解 v u 校正z的函数////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////*/
__global__ void UpdateZ(double *d,double *w,double *z,double *dev_z,int size)
{
	double temp;
	int thr=threadIdx.x;
	int blo=blockIdx.x;
	int bloDim=blockDim.x;
	int griDim=gridDim.x;
	int TF;
	for(int i=threadIdx.x+blo*bloDim;i<size;i+=bloDim*griDim)
	{
		temp=1;
		temp*=(pow(w[size-1],2)-pow(d[i],2));
		for(int j=0;j<=i-1;++j)
		{
			temp*=((pow(w[j],2)-pow(d[i],2))/(pow(d[j],2)-pow(d[i],2)));
		}
		for(int j=i;j<size-1;++j)
		{
			temp*=((pow(w[j],2)-pow(d[i],2))/(pow(d[j+1],2)-pow(d[i],2)));
		}
		TF=(z[i]>0?1:-1);
		dev_z[i]=TF*sqrt(temp);
	}
}
__global__ void adjust_z(double *d,double *w,double *z,double *dev_z,int size)
{
	double __add_left= 0;
	double __add_mid=0;
	double __add_right=0;
	double __up=0;
	double __down=0;
	int thr = threadIdx.x+blockDim.x*blockIdx.x;
	double __add ;
	for(int j=thr;j<size;j+=blockDim.x*gridDim.x)
	{
		__add_mid = 1;
		__add_left = w[size-1]*w[size-1]-d[j]*d[j];
		for(int i=0;i<j&&i<size-1;++i)
		{ 
			__up=w[i]*w[i]-d[j]*d[j];
			__down = d[i]*d[i]-d[j]*d[j];
			if(fabs(__down)<ACCURACYMIN)
			{
				__down=__down>=0?ACCURACYMIN:(0-ACCURACYMIN);
			}
			__add_mid*=(__up/__down);
		}
		__add_right = 1;
		for (int i=j;i<size-1;++i)
		{
			__up=w[i]*w[i]-d[j]*d[j];
			__down =d[i+1]*d[i+1]-d[j]*d[j];
			if(fabs(__down)<ACCURACYMIN)
			{
				__down=__down>=0?ACCURACYMIN:(0-ACCURACYMIN);
			}
			__add_right*=(__up/__down);
		}
		__add=__add_left*__add_mid*__add_right;
		__add=__add>0?__add:(0-__add);
		dev_z[j] = sqrt(__add);
		dev_z[j] = z[j]>0?dev_z[j]:(0-dev_z[j]);
	}
}
__global__ void get_u_v(double *w,double *z,double *d,double *u,double *v,int size)
{
	int thr = threadIdx.x+blockIdx.x*blockDim.x;
	double __add_u;
	double __add_v;
	double __down;
	for(int i=thr;i<size;i+=blockDim.x*gridDim.x)
	{
		__add_u = add_for_u(d,w,z,size,i);
		__add_v = add_for_v(d,w,z,size,i);
		for(int j=1;j<size;++j)
		{
			__down = d[j]*d[j]-w[i]*w[i];
			if(fabs(__down)<ACCURACYMIN)
			{
				__down = __down>=0?ACCURACYMIN:(0-ACCURACYMIN);
			}
			u[j*size+i]=z[j]/__down/__add_u;
			v[j*size+i]=z[j]*d[j]/__down/__add_v;
		}
		__down = d[0]*d[0]-w[i]*w[i];
		if(fabs(__down)<ACCURACYMIN)
		{
			__down = __down>=0?ACCURACYMIN:(0-ACCURACYMIN);
		}
		u[i]=z[0]/__down/__add_u;
		v[i]=-1/__add_v;
	}
}

__device__ double add_for_u(double *d,double *w,double *z,int size,int position)
{
	double __add=0;
	double __down;
	for(int i=0;i<size;++i)
	{
		__down = d[i]*d[i]-w[position]*w[position];
		__down = __down*__down;
		if(fabs(__down)<ACCURACYMIN)
		{
			__down = __down>0?ACCURACYMIN:(0-ACCURACYMIN);
		}
		__add+=(z[i]*z[i]/__down);
	}
	return sqrt(__add);
}

__device__  double add_for_v(double *d,double *w,double *z,int size,int position)
{
	double __add = 1.0;
	double __down;
	double __up;
	for(int i=1;i<size;++i)
	{
		__up = d[i]*z[i];
		__up = __up*__up;
		__down = d[i]*d[i]-w[position]*w[position];
		__down = __down*__down;
		if(fabs(__down)<ACCURACYMIN)
		{
			__down = __down>0?ACCURACYMIN:(0-ACCURACYMIN);
		}
		__add+=__up/__down;
	}
	return sqrt(__add);
}























/*//////////////////////////////////////////////////////////////矩阵相乘////////////////////////////////////////////////////////////*/


//__global__ void Muld_metrix(float* A, float* B, int wA, int wB, float* C)
//{ 

//	int bx = blockIdx.x; 
//	int by = blockIdx.y;    
//	int tx = threadIdx.x; 
//	int ty = threadIdx.y;    
//	int aBegin = wA * BLOCK_SIZE * by;    
//	int aEnd = aBegin + wA - 1;    
//	int aStep = BLOCK_SIZE;     
//	int bBegin = BLOCK_SIZE * bx;    
//	int bStep = BLOCK_SIZE * wB;    
//	float Csub = 0;   
//	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) 
//	{        
//		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];      
//		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];        
//		As[ty][tx] = A[a + wA * ty + tx]; 
//		Bs[ty][tx] = B[b + wB * ty + tx];        
//		__syncthreads();         
//		for (int k = 0; k < BLOCK_SIZE; ++k) 
//			Csub += As[ty][k] * Bs[k][tx];       
//		__syncthreads(); 
//	}     
//	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx; 
//	C[c + wB * ty + tx] = Csub; 
//}


bool produce_Q(GPUSave * temp,double *result,double c0,double s0)
{
	if(temp==NULL||result==NULL)
	{
		printf("Error: in the produce_Q of the secular_equation \n");
		return false;
	}
	int leMetrixSize=temp->left_size;
	int riMetrixSize=temp->right_size;
	int colNum=riMetrixSize+leMetrixSize+1;
	int rowNum=colNum+1;
	memset(result,0,sizeof(double)*colNum*rowNum);
	for(int i=0;i<leMetrixSize+1;++i)
	{
		result[i*colNum]=temp->left_q[i]*c0;
	}
	for(int i=0;i<riMetrixSize+1;++i)
	{
		result[(i+leMetrixSize+1)*colNum]=temp->right_q[i]*s0;
	}
	for(int i=0;i<leMetrixSize+1;++i)
	{
		for(int j=0;j<leMetrixSize;++j)
		{
			result[j+1+i*colNum] = temp->left_Q[j+i*leMetrixSize];
		}
	}
	for(int i=0;i<riMetrixSize+1;i++)
	{
		for (int j=0;j<riMetrixSize;++j)
		{
			result[j+leMetrixSize+1+(i+leMetrixSize+1)*colNum] = temp->right_Q[j+riMetrixSize*i];
		}
	}
	return true;
}


bool produce_W(GPUSave*temp,double *result)
{
	int first_num = temp->left_size;
	int last_num = temp->right_size;
	int size = first_num+last_num+1;
	if(temp==NULL||result==NULL)
	{
		printf("Error: in the produce_W of secular_equation\n");
		return false;
	}
	memset(result,0,sizeof(double)*size*size);
	for(int i=0;i<first_num;++i)
	{
		for(int j=0;j<first_num;++j)
		{
			result[i*size+j+1] = temp->left_W[i*first_num+j];
		}
	}
	result[first_num*size]=1;
	for(int i=0;i<last_num;++i)
	{
		for (int j=0;j<last_num;++j)
		{
			result[(i+1+first_num)*size+j+1+first_num] = temp->right_W[i*last_num+j];
		}
	}
	return true;
}

bool produce_q(GPUSave *temp,double * dev_q,double c0,double s0)
{
	int lsize=temp->left_size;
	int rsize=temp->right_size;
	double *q1=temp->left_q;
	double *q2=temp->right_q;
	for(int i=0;i<=lsize;++i)
	{
		dev_q[i]=-(s0*q1[i]);
	}
	for(int i=0;i<=rsize;i++)
	{
		dev_q[i+lsize+1]=c0*q2[i];
	}
	return true;
}
//metrix multiply function

__global__ void d_metrix_multi(double * ma, double *mb,double *mc,int wa,int wb,int ha)
{
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int he_a=by*BLOCK_SIZE+ty;
	int we_b=bx*BLOCK_SIZE+tx;
	int aBegin = BLOCK_SIZE*wa*by;
	int aEnd   = wa;
	int aStep  = BLOCK_SIZE;
	int bBegin = BLOCK_SIZE*bx;
	int bStep  = BLOCK_SIZE*wb;
	__shared__ double da[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ double db[BLOCK_SIZE][BLOCK_SIZE];
	double csum=0;
	int i,j;
	for(i=0,j=0; i<aEnd ; i+=aStep,j+=bStep)
	{
		if(he_a<ha&&(tx+i)<wa)
		{
			da[ty][tx] = ma[i+aBegin+tx+ty*wa];
		}
		else
		{
			da[ty][tx] = 0;
		}
		if(we_b<wb&&(i+ty)<wa)
		{
			db[ty][tx] = mb[j+bBegin+tx+ty*wb];
		}
		else
		{
			db[ty][tx] = 0;
		}
		__syncthreads();

		for(int k=0;k<BLOCK_SIZE;k++)
		{
			csum+=da[ty][k]*db[k][tx];
		}
		__syncthreads();
	}
	if(he_a<ha&&we_b<wb)
	{
		mc[(ty+by*BLOCK_SIZE)*wb+tx+BLOCK_SIZE*bx]=csum;
	}
}
	





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