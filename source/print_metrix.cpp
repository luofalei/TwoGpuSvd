#include <stdio.h>
#include	"print_metrix.h" 
#include <stdlib.h>
#include <math.h>
#include "first_svd.h"
bool print_mid_metrix(FILE *ph,double *Q,double *q,double *D,double *W,int size)
{
	fprintf(ph,"the Q shows:\n");
	print_block(ph,Q,size+1,size);
	fprintf(ph,"the q shows:\n");
	print_block(ph,q,size+1,1);
	fprintf(ph,"the D shows:\n");
	print_block(ph,D,1,size);
	fprintf(ph,"the W shows:\n");
	print_block(ph,W,size,size);
	return true;
}

bool print_block(FILE *ph,double * block, int rows,int cols)
{
	for(int i=0;i<rows;++i){
		for(int j=0;j<cols;++j){
			fprintf(ph,"%lf ",*(block+i*cols+j));
		}
		fprintf(ph,"\n");
	}
	return true;
}
bool print_original_metrix(FILE*ph,double* diagonal,double *offdiagonal,int size)
{
	fprintf(ph,"the diagonal shows :\n");
	print_block(ph,diagonal,size,1);
	fprintf(ph,"the offdiagonal shows:\n");
	print_block(ph,offdiagonal,size,1);
	return true;
}

bool print_svd_metrix(FILE *ph,double *X,double *w,double *Y,int size)
{
	fprintf(ph,"the X shows :\n");
	print_block(ph,X,size+1,size+1);
	fprintf(ph,"the w shows :\n");
	print_block(ph,w,1,size);
	fprintf(ph,"the Y shows:\n");
	print_block(ph,Y,size,size);
	return true;
}

bool reCheack(FILE *ph,double *rQ,double *rq,double *rD,double *rW,int size)
{
//	double *cQ,*cq,*cD,*cW;
//	cQ=MASIZE(double,size*(size+1));
//	cq=MASIZE(double,size+1);
//	cD=MASIZE(double,size);
//	cW=MASIZE(double,size*size);
//	double *biognalUp,*biognalDown;
//	biognalUp=MASIZE(double,size);
//	biognalDown=MASIZE(double,size);
//	for(int i=1;i<=size;++i)
//	{
//		biognalUp[i-1]=1.0*i;
//		biognalDown[i-1]=2.0*i;
//	}
//	gpu_svd::first_svd(biognalUp,biognalDown,size,cQ,cq,cD,cW);
	print_mid_metrix(ph,rQ,rq,rD,rW,size);
//	fprintf(ph,"\n\n\n\n\n\n\n");
//	print_mid_metrix(ph,cQ,cq,cD,cW,size);
//	for(int i=0;i<size+1;++i)
//	{
//		for(int j=0;j<size;++j)
//			cQ[i*size+j]=cQ[i*size+j]-rQ[i*size+j];
//	}
//	for(int i=0;i<size+1;i++)
//	{
//		cq[i]=cq[i]-rq[i];
//	}
//	for(int i=0;i<size;++i)
//	{
//		cD[i]=cD[i]-rD[i];
//	}
//	for(int i=0;i<size;++i)
//	{
//		for(int j=0;j<size;++j)
//		{
//			cW[i*size+j]=cW[i*size+j]-rW[i*size+j];
//		}
//	}
//	print_mid_metrix(ph,cQ,cq,cD,cW,size);
	return true;
}

