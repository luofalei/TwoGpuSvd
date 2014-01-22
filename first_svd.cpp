#include "first_svd.h"
#include "print_metrix.h"
#include <stdlib.h>
#include <linalg.h>

/***************************************************************************************
 * 下面是该类的具体实现
 * 功能只是第一次svd分解
 ***************************************************************************************/
using namespace gpu_svd;
/**************************************************************************************************
 *用alglib库进行奇异值分解
 *该程序是在cpu上进行的
 *_m行数，_n列数, _uneed _vneed _speed 不用考虑，这存在问题，是可以优化的，考虑是否能直接svd双对角的
 *************************************************************************************************/
bool gpu_svd::first_svd(double *diagonal,double *offdiagonal,int size,double *Q,double *q,double *D,double *W)
{
	alglib::real_2d_array a;
	alglib::ae_int_t _m=size+1;
	alglib::ae_int_t _n = size;
	alglib::ae_int_t _uneed = 2;
	alglib::ae_int_t _vneed = 2;
	alglib::ae_int_t _speed = 0;
	alglib::real_2d_array _u;
	alglib::real_2d_array _vt;
	alglib::real_1d_array _w;
	double *_metrix;
	_metrix = (double *)malloc(sizeof(double)*size*(size+1));
	if (diagonal==NULL||offdiagonal==NULL)
	{
		printf("error in the fist_svd function");
		return 0;
	}
	for(int i=0;(unsigned int)i<size+1;i++)
	{
		for(int j=0;j<size;++j)
		{
			if(i==j)
			{
				*(_metrix+i*size+j)=diagonal[i];
			}
			else if(i==(j+1))
			{
				*(_metrix+i*size+j)=offdiagonal[j];
			}
			else
			{
				*(_metrix+i*size+j)=0;
			}
		}
	}
	a.setcontent(_m,_n,_metrix);

	alglib::rmatrixsvd(a,_m,_n,_uneed,_vneed,_speed,_w,_u,_vt);
	
	gpu_svd::get_1d_fromalg(_w.length(),D,&_w);                    /*get the D*/
	gpu_svd::get_2d_fromalg(0,_u.rows(),0,_u.cols()-1,Q,&_u);  /*get the Q*/
	gpu_svd::get_2d_fromalg(0,_u.rows(),_u.cols()-1,1,q,&_u);  /*get the q*/
	gpu_svd::transpose(_vt.rows(),_vt.cols(),W,&_vt);
	//gpu_svd::increase(size,Q,q,D,W);
	return true;
}
bool gpu_svd::get_1d_fromalg(int col,double *metrix,alglib::real_1d_array *a)
{
	if(a->length()<col)
	{
		return 0;
	}
	for(int i=0;i<col;++i)
	{
		metrix[i]=(*a)[i];
	}
	return 1;
}


bool gpu_svd::get_2d_fromalg( int start_row,int row_num,int start_col,int col_num,double *metrix,alglib::real_2d_array *a)
{
	if(a->cols()<start_col+col_num||a->rows()<start_row+row_num)
	{
		return 0;
	}
	for(int i=0;i<row_num;i++)
	{
		for(int j=0;j<col_num;++j)
		{
			*(metrix+i*col_num+j)=(*a)[i+start_row][start_col+j];
		}
	}
	return 1;
}
bool gpu_svd::transpose(int rowNum,int colNum,double *metrix,alglib::real_2d_array *a)
{
	for(int i=0;i<rowNum;++i)
	{
		for(int j=0;j<colNum;++j)
		{
			*(metrix+j*rowNum+i) = (*a)[i][j];
		}
	}
	return 1;
}



bool gpu_svd::increase(int size,double *Q,double *q,double *d,double *w)
{

	FILE *phte=fopen("F:/temetrix.txt","w");
	print_mid_metrix(phte,Q,q,d,w,size);
	fclose(phte);

	double temp;
	for(int i=0;i<size/2;++i)
	{
		for(int j=0;j<size;++j)
		{
			temp=Q[j*size+i];
			Q[j*size+i]=Q[j*size+size-1-i];
			Q[j*size+size-1-i]=temp;
			temp=w[i*size+j];
			w[i*size+j]=w[(size-1-i)*size+j];
			w[(size-1-i)*size+j]=temp;
		}
		temp=Q[size*size+i];
		Q[size*size+i]=Q[size*size+size-1-i];
		Q[size*size+size-1-i]=temp;
		temp=d[i];
		d[i]=d[size-1-i];
		d[size-1-i]=temp;
	}
//		temp=q[i];
//		q[i]=q[size-1-i];
//		q[size-i-1]=temp;
//	if(!size%2)
//	{
//		int i=size/2;
//		for(int j=0;j<size;++j)
//		{
//			temp=Q[i*size+j];
//			Q[i*size+j]=Q[(size-1-i)*size+j];
//			Q[(size-i-1)*size+j]=temp;
//			temp=w[j*size+i];
//			w[j*size+i]=w[j*size+size-1-i];
//			w[j*size+size-1-i]=temp;
//		}
//		temp=d[i];
//		d[i]=d[size-1-i];
//		d[size-1-i]=temp;
//		temp=q[i];
//		q[i]=q[size-1-i];
//		q[size-i-1]=temp;
//	}

	


//	FILE *ph=fopen("F:/metrixBiogonal.txt","w");
//	gpu_svd::reMulti(ph,size,Q,d,w);
//	fclose(ph);
	return true;
}

bool gpu_svd::reMulti(FILE *ph,int size,double *Q ,double *D, double *W)
{
	double *reMetrix=(double *)malloc(sizeof(double)*size*(size+1));
	double *result=(double *)malloc(sizeof(double)*size*(size+1));
	double temp;
	double *devD=(double *)malloc(sizeof(double)*size*size);
	for(int i=0;i<size;++i)
	{
		for(int j=0;j<i;++j)
		{
			devD[i*size+j]=0;
		}
		devD[i*size+i]=D[i];

		for(int j=i+1;j<size;++j)
		{
			devD[i*size+j]=0;
		}
	}

		
	for(int i=0;i<size+1;++i)
	{
		for(int j=0;j<size;++j)
		{
			temp=0;
			for(int k=0;k<size;++k)
			{
				temp+=Q[i*size+k]*devD[k*size+j];
			}
			reMetrix[i*size+j]=temp;
		}
			
	}
	for(int i=0;i<size+1;++i)
	{
		for(int j=0;j<size;++j)
		{
			temp=0;
			for(int k=0;k<size;++k)
			{
				temp+=reMetrix[i*size+k]*W[k*size+j];
			}
			result[i*size+j]=temp;
		}
	}
	print_block(ph,result,size+1,size);
	return 0;
}
		

