#include "GpuMultiplyMetrix.cuh"
void metrixMultiply(PMetrix a,PMetrix b ,PMetrix c)
{
	double *dev_a,*dev_b,*dev_c;
	double alpha = 1.0;
	const double bate  = 0.0;
	cublasInit();
	cublasAlloc(a->rows*a->cols,sizeof(double),(void**)&dev_a);
	cublasAlloc(b->rows*b->cols,sizeof(double),(void**)&dev_b);
	cublasAlloc(c->rows*c->cols,sizeof(double),(void**)&dev_c);
	cudaMemcpy(dev_a,a->data,sizeof(double)*a->cols*a->rows,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,b->data,sizeof(double)*b->cols*b->rows,cudaMemcpyHostToDevice);
	cublasHandle_t  handles;
	cublasCreate(&handles);
	cublasDgemm(handles,CUBLAS_OP_N,CUBLAS_OP_N,a->rows,b->cols,b->rows,&alpha,dev_a,a->rows,dev_b,b->rows,&bate,dev_c,a->rows);
	cudaMemcpy(c->data,dev_c,sizeof(double)*c->cols*c->rows,cudaMemcpyDeviceToHost);
	cublasShutdown();
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}
