#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GpuMultiplyMetrix.cuh"
#include <helper_cuda.h>
#include <device_functions.h>
#include "devide_metrix.h"
#include <stdio.h>
#include <math.h>

#include "sequanceEquation2.cuh"
#include "sequanceEquation.cuh"
/*������*/

bool   produce_mid_metrix(GPUSave *temp,int DeviceID,bool result_type);





/*������������������м�����*/
bool  formM(double alpha,double beta,double *Q1,double *q1,\
	double *D1,double *Q2,double *q2,double *D2,double *z,double *d,\
	int up_number,int down_number,double &gama);

bool  sortM(int n,double *z,double *d,double *znew,double*dnew);
/*sort z[n] and d[n] of Matrix M, and d[n] adds another element for the secular equation*/

/*������������w���м������Ŀ���ǿ�������Բ���*/

/*�����м亯��*/
bool mid_function(double *mid_d,double *mid_z,double *w,int size);

bool mid_function_two(double *temp_d,double * temp_z,double *temp_w,double *Q,double *W,\
	double *result_Q,double *result_W,int size);


double geometric_z(double *mid_z,int size);






bool deThread(int size,int &thread,int &block);

void mergeSort(double *Data,int size,double *z);
void merge(double *left,int leftSize,double *right,int rightSize,double *Zleft,double *Zright);

/*���������������͵�*/
__device__  double  add_secular(double *temp_d,double *temp_z,int size,double w);


/*����������������ַ���*/
/*������Ǹ���ƽ����ϵ������������ٶȣ�w2,z2,d2*/

__device__ double  GPU_FindOneRoot_Bisection(double  *d, double  *z, int  len_d_z, double  Ulimit, double  Dlimit,int  &IsSuccess,double &temps);


/*�����������������ڷ��̵�*/
__global__ void secular_equation_small(double *z,double *d,double *w,int size,double *temps,double *,double *);



/*/////////////////////////////////////////////////////////У��z,���u,v//////////////////////////////////////////////////////////////*/
__device__  double add_for_v(double *d,double *w,double *z,int size,int position);
__device__ double add_for_u(double *d,double *w,double *z,int size,int position);
__global__ void get_u_v(double *w,double *z,double *d,double *u,double *v,int size);
__global__ void adjust_z(double *d,double *w,double *z,double *dev_z,int size);




/***********************************�������************************************************************/
bool produce_Q(GPUSave * temp,double *result,double c0,double s0);
bool produce_W(GPUSave*temp,double *result);
bool produce_q(GPUSave*temp,double *result,double c0,double s0);

/***********************************�������***************************************************************/
__global__ void d_metrix_multi(double * ma, double *mb,double *mc,int wa,int wb,int ha);
	
__global__ void UpdateZ(double *d,double *w,double *z,double *dev_z,int size);




__global__ void Solve(double *z,double *d,double *sigma,int size);
__device__ double sequanceSolve(double *z,double *d,double DLimit,double ULimit,int size);
__device__  double add(double *z,double *d,double sigma,int size);
void SequanceEquationSolve(double *z,double *d,double *sigma,int size);











double * mulMetrixSove(double *Q,double *W,int lcol,int lrow,int rcol);
double * decode(double *Q,int cols,int rows);
double * encode(double *Q,int cols,int rows);