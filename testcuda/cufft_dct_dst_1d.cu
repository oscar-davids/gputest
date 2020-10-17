#include "cufft.h"
#ifndef pi
#define pi 4.0f*atanf(1.0f)
#endif
#ifndef threads_num
#define threads_num 256
#endif

/* functions for dst and dct transforms */
static __global__ void DataExtSetRedft00_1d(float *d_in,  cufftComplex *d_out, int n);
static __global__ void DataSubGetRedft00_1d(cufftComplex *d_in, float *d_out, int n);
static __global__ void DataExtSetRedft10_1d(float *d_in,  cufftComplex *d_out, int n);
static __global__ void DataSubGetRedft10_1d(cufftComplex *d_in, float *d_out, int n);
static __global__ void DataExtSetRedft01_1d(float *d_in,  cufftComplex *d_out, int n);
static __global__ void DataSubGetRedft01_1d(cufftComplex *d_in, float *d_out, int n);
static __global__ void DataExtSetRodft00_1d(float *d_in,  cufftComplex *d_out, int n);
static __global__ void DataSubGetRodft00_1d(cufftComplex *d_in, float *d_out, int n);
static __global__ void DataExtSetIRodft00_1d(float *d_in,  cufftComplex *d_out, int n);
static __global__ void DataSubGetIRodft00_1d(cufftComplex *d_in, float *d_out, int n);
static __global__ void DataExtSetRodft10_1d(float *d_in,  cufftComplex *d_out, int n);
static __global__ void DataSubGetRodft10_1d(cufftComplex *d_in, float *d_out, int n);
static __global__ void DataExtSetRodft01_1d(float *d_in,  cufftComplex *d_out, int n);
static __global__ void DataSubGetRodft01_1d(cufftComplex *d_in, float *d_out, int n);
static __global__ void PhaseShiftForw_1d(cufftComplex *d_in, cufftComplex *d_out, int n);
static __global__ void PhaseShiftBack_1d(cufftComplex *d_in, cufftComplex *d_out, int n);

/* various dst and dct transforms */
void cufft_redft00_1d(float *d_in, float *d_out, int n);
void cufft_iredft00_1d(float *d_in, float *d_out, int n);
void cufft_redft10_1d(float *d_in, float *d_out, int n);
void cufft_redft01_1d(float *d_in, float *d_out, int n);
void cufft_rodft00_1d(float *d_in, float *d_out, int n);
void cufft_irodft00_1d(float *d_in, float *d_out, int n);
void cufft_rodft10_1d(float *d_in, float *d_out, int n);
void cufft_rodft01_1d(float *d_in, float *d_out, int n);

/* functions for first-order derivatives computed by dst and dct */
static __global__ void dst1_idct2_1d_pre(float *d_in, float *d_out, float dx, int n);
static __global__ void dst1_idct2_1d_post(float *d_in, float *d_out, int n);
static __global__ void dct1_idst2_1d_pre(float *d_in, float *d_out, float dx, int n);
static __global__ void dct1_idst2_1d_post(float *d_in, float *d_out, int n);
static __global__ void dst2_idct1_1d_pre(float *d_in, float *d_out, float dx, int n);
static __global__ void dst2_idct1_1d_post(float *d_in, float *d_out, int n);
static __global__ void dct2_idst1_1d_pre(float *d_in, float *d_out, float dx, int n);
static __global__ void dct2_idst1_1d_post(float *d_in, float *d_out, int n);

void cufft_dct2_idst1_1d(float *d_in, float *d_out, float dx, int nn)
{
	int n = nn-1;
	
	float *d_in_dst;
	cudaMalloc((void **)&d_in_dst, n*sizeof(float));
	
	/* Sine transform */
	cufft_redft10_1d(d_in, d_in_dst, n);
	
	float *d_in_tmp;
	cudaMalloc((void **)&d_in_tmp, (n-1)*sizeof(float));
	
	/* Multiplied by wavenumber */
	dct2_idst1_1d_pre<<<(n+threads_num-1)/threads_num, threads_num>>>(d_in_dst, d_in_tmp, dx, n);
	
	float *d_out_tmp;
	cudaMalloc((void **)&d_out_tmp, (n-1)*sizeof(float));
	/* Inverse cosine transform */
	cufft_irodft00_1d(d_in_tmp, d_out_tmp, n-1);

	/* Get the result */
	dct2_idst1_1d_post<<<(n+threads_num-1)/threads_num, threads_num>>>(d_out_tmp, d_out, n);

	cudaFree(d_in_dst);
	cudaFree(d_in_tmp);
	cudaFree(d_out_tmp);
}

static __global__ void dct2_idst1_1d_pre(float *d_in, float *d_out, float dx, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	d_out[0] = 0.0f;
	//for (int ix=tid; ix<n-1; ix+=threads_num)
	if (ix < n-1)
		d_out[ix] = -(float)(ix+1)*pi/((float)n*dx)*d_in[ix+1];
}

static __global__ void dct2_idst1_1d_post(float *d_in, float *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	//for (int ix=tid; ix<n-1; ix+=threads_num)
	if (ix < n-1)
		d_out[ix+1] = d_in[ix]/(2.0f*(float)(n));
	d_out[n] = 0.0f;
}

void cufft_dst2_idct1_1d(float *d_in, float *d_out, float dx, int nn)
{
	int n = nn-1;
	
	float *d_in_dst;
	cudaMalloc((void **)&d_in_dst, n*sizeof(float));
	
	/* Sine transform */
	cufft_rodft10_1d(d_in, d_in_dst, n);
	
	float *d_in_tmp;
	cudaMalloc((void **)&d_in_tmp, (n+1)*sizeof(float));
	
	/* Multiplied by wavenumber */
	dst2_idct1_1d_pre<<<(n+threads_num-1)/threads_num, threads_num>>>(d_in_dst, d_in_tmp, dx, n);
	
	float *d_out_tmp;
	cudaMalloc((void **)&d_out_tmp, (n+1)*sizeof(float));
	/* Inverse cosine transform */
	cufft_redft00_1d(d_in_tmp, d_out_tmp, n+1);

	/* Get the result */
	dst2_idct1_1d_post<<<(n+threads_num-1)/threads_num, threads_num>>>(d_out_tmp, d_out, n);

	cudaFree(d_in_dst);
	cudaFree(d_in_tmp);
	cudaFree(d_out_tmp);
}

static __global__ void dst2_idct1_1d_pre(float *d_in, float *d_out, float dx, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	d_out[0] = 0.0f;
	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
		d_out[ix+1] = (float)(ix+1)*pi/((float)n*dx)*d_in[ix];
}

static __global__ void dst2_idct1_1d_post(float *d_in, float *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	//for (int ix=tid; ix<n+1; ix+=threads_num)
	if (ix < n+1)
		d_out[ix] = d_in[ix]/(2.0f*(float)(n));
}

void cufft_dct1_idst2_1d(float *d_in, float *d_out, float dx, int n)
{
	float *d_in_dst;
	cudaMalloc((void **)&d_in_dst, n*sizeof(float));
	
	/* Sine transform */
	cudaMemset((void *)&d_in[n-1], 0, sizeof(int));
	cufft_redft00_1d(d_in, d_in_dst, n);
	
	float *d_in_tmp;
	cudaMalloc((void **)&d_in_tmp, (n-1)*sizeof(float));
	
	/* Multiplied by wavenumber */
	dct1_idst2_1d_pre<<<(n+threads_num-1)/threads_num, threads_num>>>(d_in_dst, d_in_tmp, dx, n);
	
	float *d_out_tmp;
	cudaMalloc((void **)&d_out_tmp, (n-1)*sizeof(float));
	/* Inverse cosine transform */
	cufft_rodft01_1d(d_in_tmp, d_out_tmp, n-1);

	/* Get the result */
	dct1_idst2_1d_post<<<(n+threads_num-1)/threads_num, threads_num>>>(d_out_tmp, d_out, n);

	cudaFree(d_in_dst);
	cudaFree(d_in_tmp);
	cudaFree(d_out_tmp);
}

static __global__ void dct1_idst2_1d_pre(float *d_in, float *d_out, float dx, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	d_out[0] = 0.0f;
	//for (int ix=tid; ix<n-1; ix+=threads_num)
	if (ix < n-1)
		d_out[ix] = -(float)(ix+1)*pi/(float(n-1)*dx)*d_in[ix+1];
}

static __global__ void dct1_idst2_1d_post(float *d_in, float *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	//for (int ix=tid; ix<n-1; ix+=threads_num)
	if (ix < n-1)
		d_out[ix] = d_in[ix]/(2.0f*(float)(n-1));
	d_out[n-1] = 0.0f;
}

void cufft_dst1_idct2_1d(float *d_in, float *d_out, float dx, int n)
{
	float *d_in_dst;
	cudaMalloc((void **)&d_in_dst, n*sizeof(float));
	
	/* Sine transform */
	cufft_rodft00_1d(d_in, d_in_dst, n);
	
	float *d_in_tmp;
	cudaMalloc((void **)&d_in_tmp, (n+1)*sizeof(float));
	
	/* Multiplied by wavenumber */
	dst1_idct2_1d_pre<<<(n+threads_num-1)/threads_num, threads_num>>>(d_in_dst, d_in_tmp, dx, n);
	
	float *d_out_tmp;
	cudaMalloc((void **)&d_out_tmp, (n+1)*sizeof(float));
	/* Inverse cosine transform */
	cufft_redft01_1d(d_in_tmp, d_out_tmp, n+1);

	/* Get the result */
	dst1_idct2_1d_post<<<(n+threads_num-1)/threads_num, threads_num>>>(d_out_tmp, d_out, n);

	cudaFree(d_in_dst);
	cudaFree(d_in_tmp);
	cudaFree(d_out_tmp);
}

static __global__ void dst1_idct2_1d_pre(float *d_in, float *d_out, float dx, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
		d_out[ix+1] = (float)(ix+1)*pi/(float(n+1)*dx)*d_in[ix];
	d_out[0] = 0.0f;
}

static __global__ void dst1_idct2_1d_post(float *d_in, float *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	//for (int ix=tid; ix<n-1; ix+=threads_num)
	if (ix < n-1)
		d_out[ix] = d_in[ix+1]/(2.0f*(float)(n+1));
	d_out[n-1] = 0.0f;
}

void cufft_redft00_1d(float *d_in, float *d_out, int n)
{
	/* Allocate memory for extended complex data */
	cufftComplex *d_in_ext;
	cudaMalloc((void **)&d_in_ext, (2*n-2)*sizeof(cufftComplex));

	/* Extend and convert the data from float to cufftComplex */
	DataExtSetRedft00_1d<<<(n+threads_num-1)/threads_num, threads_num>>>(d_in, d_in_ext, n);
	
	/* Create a 1D FFT plan */
	cufftHandle plan;
	cufftPlan1d(&plan, 2*n-2, CUFFT_C2C, 1);

	/* Allocate memory for transformed data */
	cufftComplex *d_out_ext;
	cudaMalloc((void **)&d_out_ext, (2*n-2)*sizeof(cufftComplex));
	
	/* Use the CUFFT plan to transform the signal out of place */
	cufftExecC2C(plan, d_in_ext, d_out_ext, CUFFT_FORWARD);

	/* Subtract the transformed data */
	DataSubGetRedft00_1d<<<(n+threads_num-1)/threads_num, threads_num>>>(d_out_ext, d_out, n);

	cufftDestroy(plan);
	cudaFree(d_in_ext);
	cudaFree(d_out_ext);
}

void cufft_iredft00_1d(float *d_in, float *d_out, int n)
{
	cufft_redft00_1d(d_in, d_out, n);
}

void cufft_redft10_1d(float *d_in, float *d_out, int n)
{	
	/* Allocate meory for extended complex data */
	cufftComplex *d_in_ext;
	cudaMalloc((void **)&d_in_ext, 2*n*sizeof(cufftComplex));

	/* Extend and convert the data from float to cufftComplex */
	DataExtSetRedft10_1d<<<(n+threads_num-1)/threads_num, threads_num>>>(d_in, d_in_ext, n);

	/* Create a 1D FFT plan */
	cufftHandle plan;
	cufftPlan1d(&plan, 2*n, CUFFT_C2C, 1);

	/* Allocate memory for transformed data */
	cufftComplex *d_out_ext;
	cudaMalloc((void **)&d_out_ext, 2*n*sizeof(cufftComplex));
	
	/* Use the CUFFT plan to transform the signal out of place */
	cufftExecC2C(plan, d_in_ext, d_out_ext, CUFFT_FORWARD);

	/* Allocate memory for shifted data */
	cufftComplex *d_out_ext_shift;
	cudaMalloc((void **)&d_out_ext_shift, 2*n*sizeof(cufftComplex));
	
	/* 1/2 phase shift */
	PhaseShiftForw_1d<<<(2*n+threads_num-1)/threads_num, threads_num>>>(d_out_ext, d_out_ext_shift, 2*n);

	/* Subtract the transformed data */
	DataSubGetRedft10_1d<<<(n+threads_num-1)/threads_num, threads_num>>>(d_out_ext_shift, d_out, n);

	cufftDestroy(plan);
	cudaFree(d_in_ext);
	cudaFree(d_out_ext);
	cudaFree(d_out_ext_shift);
}

void cufft_redft01_1d(float *d_in, float *d_out, int n)
{
	/* Allocate meory for extended complex data */
	cufftComplex *d_in_ext;
	cudaMalloc((void **)&d_in_ext, 2*n*sizeof(cufftComplex));
	
	/* Extend and convert the data from float to cufftComplex */
	DataExtSetRedft01_1d<<<(n+threads_num-1)/threads_num, threads_num>>>(d_in, d_in_ext, n);

	/* Allocate memory for shifted data */
	cufftComplex *d_in_ext_shift;
	cudaMalloc((void **)&d_in_ext_shift, 2*n*sizeof(cufftComplex));
	
	/* -1/2 phase shift  */
	PhaseShiftBack_1d<<<(2*n+threads_num-1)/threads_num, threads_num>>>(d_in_ext, d_in_ext_shift, 2*n);

	/* Create a 1D FFT plan */
	cufftHandle plan;
	cufftPlan1d(&plan, 2*n, CUFFT_C2C, 1);

	/* Allocate memory for transformed data */
	cufftComplex *d_out_ext;
	cudaMalloc((void **)&d_out_ext, 2*n*sizeof(cufftComplex));
	
	/* Use the CUFFT plan to transform the signal out of place */
	cufftExecC2C(plan, d_in_ext_shift, d_out_ext, CUFFT_INVERSE);

	/* Subtract the transformed data */
	DataSubGetRedft01_1d<<<(n+threads_num-1)/threads_num, threads_num>>>(d_out_ext, d_out, n);

	cufftDestroy(plan);
	cudaFree(d_in_ext);
	cudaFree(d_in_ext_shift);
	cudaFree(d_out_ext);
}

void cufft_rodft00_1d(float *d_in, float *d_out, int n)
{
	/* Allocate memory for extended complex data */
	cufftComplex *d_in_ext;
	cudaMalloc((void **)&d_in_ext, (2*n+2)*sizeof(cufftComplex));

	/* Extend and convert the data from float to cufftComplex */
	DataExtSetRodft00_1d<<<(n+threads_num-1)/threads_num, threads_num>>>(d_in, d_in_ext, n);
	
	/* Create a 1D FFT plan */
	cufftHandle plan;
	cufftPlan1d(&plan, 2*n+2, CUFFT_C2C, 1);

	/* Allocate memory for transformed data */
	cufftComplex *d_out_ext;
	cudaMalloc((void **)&d_out_ext, (2*n+2)*sizeof(cufftComplex));
	
	/* Use the CUFFT plan to transform the signal out of place */
	cufftExecC2C(plan, d_in_ext, d_out_ext, CUFFT_FORWARD);

	/* Subtract the transformed data */
	DataSubGetRodft00_1d<<<(n+threads_num-1)/threads_num, threads_num>>>(d_out_ext, d_out, n);

	cufftDestroy(plan);
	cudaFree(d_in_ext);
	cudaFree(d_out_ext);
}

void cufft_irodft00_1d(float *d_in, float *d_out, int n)
{
	/* Allocate memory for extended complex data */
	cufftComplex *d_in_ext;
	cudaMalloc((void **)&d_in_ext, (2*n+2)*sizeof(cufftComplex));

	/* Extend and convert the data from float to cufftComplex */
	DataExtSetIRodft00_1d<<<(n+threads_num-1)/threads_num, threads_num>>>(d_in, d_in_ext, n);

	/* Create a 1D FFT plan */
	cufftHandle plan;
	cufftPlan1d(&plan, 2*n+2, CUFFT_C2C, 1);

	/* Allocate memory for transformed data */
	cufftComplex *d_out_ext;
	cudaMalloc((void **)&d_out_ext, (2*n+2)*sizeof(cufftComplex));
	
	/* Use the CUFFT plan to transform the signal out of place */
	cufftExecC2C(plan, d_in_ext, d_out_ext, CUFFT_INVERSE);

	/* Subtract the transformed data */
	DataSubGetIRodft00_1d<<<(n+threads_num-1)/threads_num, threads_num>>>(d_out_ext, d_out, n);

	cufftDestroy(plan);
	cudaFree(d_in_ext);
	cudaFree(d_out_ext);
}

void cufft_rodft10_1d(float *d_in, float *d_out, int n)
{
	/* Allocate meory for extended complex data */
	cufftComplex *d_in_ext;
	cudaMalloc((void **)&d_in_ext, 2*n*sizeof(cufftComplex));

	/* Extend and convert the data from float to cufftComplex */
	DataExtSetRodft10_1d<<<(n+threads_num-1)/threads_num, threads_num>>>(d_in, d_in_ext, n);

	/* Create a 1D FFT plan */
	cufftHandle plan;
	cufftPlan1d(&plan, 2*n, CUFFT_C2C, 1);

	/* Allocate memory for transformed data */
	cufftComplex *d_out_ext;
	cudaMalloc((void **)&d_out_ext, 2*n*sizeof(cufftComplex));
	
	/* Use the CUFFT plan to transform the signal out of place */
	cufftExecC2C(plan, d_in_ext, d_out_ext, CUFFT_FORWARD);

	/* Allocate memory for shifted data */
	cufftComplex *d_out_ext_shift;
	cudaMalloc((void **)&d_out_ext_shift, 2*n*sizeof(cufftComplex));
	
	/* 1/2 phase shift */
	PhaseShiftForw_1d<<<(2*n+threads_num-1)/threads_num, threads_num>>>(d_out_ext, d_out_ext_shift, 2*n);

	/* Subtract the transformed data */
	DataSubGetRodft10_1d<<<(n+threads_num-1)/threads_num, threads_num>>>(d_out_ext_shift, d_out, n);

	cufftDestroy(plan);
	cudaFree(d_in_ext);
	cudaFree(d_out_ext);
	cudaFree(d_out_ext_shift);
}

void cufft_rodft01_1d(float *d_in, float *d_out, int n)
{
	/* Allocate meory for extended complex data */
	cufftComplex *d_in_ext;
	cudaMalloc((void **)&d_in_ext, 2*n*sizeof(cufftComplex));
	
	/* Extend and convert the data from float to cufftComplex */
	DataExtSetRodft01_1d<<<(n+threads_num-1)/threads_num, threads_num>>>(d_in, d_in_ext, n);
	
	/* Allocate memory for shifted data */
	cufftComplex *d_in_ext_shift;
	cudaMalloc((void **)&d_in_ext_shift, 2*n*sizeof(cufftComplex));
	
	/* -1/2 phase shift  */
	PhaseShiftBack_1d<<<(2*n+threads_num-1)/threads_num, threads_num>>>(d_in_ext, d_in_ext_shift, 2*n);

	/* Create a 1D FFT plan */
	cufftHandle plan;
	cufftPlan1d(&plan, 2*n, CUFFT_C2C, 1);

	/* Allocate memory for transformed data */
	cufftComplex *d_out_ext;
	cudaMalloc((void **)&d_out_ext, 2*n*sizeof(cufftComplex));
	
	/* Use the CUFFT plan to transform the signal out of place */
	cufftExecC2C(plan, d_in_ext_shift, d_out_ext, CUFFT_INVERSE);

	/* Subtract the transformed data */
	DataSubGetRodft01_1d<<<(n+threads_num-1)/threads_num, threads_num>>>(d_out_ext, d_out, n);

	cufftDestroy(plan);
	cudaFree(d_in_ext);
	cudaFree(d_in_ext_shift);
	cudaFree(d_out_ext);
}

static __global__ void DataExtSetRedft00_1d(float *d_in,  cufftComplex *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	
	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
	{
		d_out[ix].x = d_in[ix];
		d_out[ix].y = 0.0f;
		if (ix<n-2)
		{
			d_out[2*n-3-ix].x = d_in[ix+1];
			d_out[2*n-3-ix].y = 0.0f;
		}
	}
}

static __global__ void DataSubGetRedft00_1d(cufftComplex *d_in, float *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	
	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)	
		d_out[ix] = d_in[ix].x;
}

static __global__ void DataExtSetRedft10_1d(float *d_in, cufftComplex *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
	{
		d_out[ix].x = d_in[ix];
		d_out[ix].y = 0.0f;
		d_out[2*n-1-ix].x = d_in[ix];
		d_out[2*n-1-ix].y = 0.0f;
	}
}

static __global__ void DataSubGetRedft10_1d(cufftComplex *d_in, float *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	
	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
		d_out[ix] = d_in[ix].x;
}

static __global__ void DataExtSetRedft01_1d(float *d_in, cufftComplex *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
	{
		d_out[ix].x = d_in[ix];
		d_out[ix].y = 0.0f;
		if (ix<n-1)
		{
			d_out[2*n-1-ix].x = d_in[ix+1];
			d_out[2*n-1-ix].y = 0.0f;
		}
	}
}

static __global__ void DataSubGetRedft01_1d(cufftComplex *d_in, float *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
		d_out[ix] = d_in[ix].x;
}

static __global__ void DataExtSetRodft00_1d(float *d_in, cufftComplex *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
	{
		d_out[ix+1].x = d_in[ix];
		d_out[ix+1].y = 0.0f;
		d_out[2*n+1-ix].x = -d_in[ix];
		d_out[2*n+1-ix].y = 0.0f;
	}
	d_out[0].x = d_out[0].y = 0.0f;
	d_out[n+1].x = d_out[n+1].y = 0.0f;
}

static __global__ void DataSubGetRodft00_1d(cufftComplex *d_in, float *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
		d_out[ix] = -d_in[ix+1].y;
}

static __global__ void DataExtSetIRodft00_1d(float *d_in, cufftComplex *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
	{
		d_out[ix+1].y = -d_in[ix];
		d_out[ix+1].x = 0.0f;
		d_out[2*n+1-ix].y = d_in[ix];
		d_out[2*n+1-ix].x = 0.0f;
	}
	d_out[0].x = d_out[0].y = 0.0f;
	d_out[n+1].x = d_out[n+1].y = 0.0f;
}

static __global__ void DataSubGetIRodft00_1d(cufftComplex *d_in, float *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
		d_out[ix] = d_in[ix+1].x;
}

static __global__ void DataExtSetRodft10_1d(float *d_in, cufftComplex *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
	{
		d_out[ix].x = d_in[ix];
		d_out[ix].y = 0.0f;
		d_out[2*n-1-ix].x = -d_in[ix];
		d_out[2*n-1-ix].y = 0.0f;
	}
}

static __global__ void DataSubGetRodft10_1d(cufftComplex *d_in, float *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
		d_out[ix] = -d_in[ix+1].y;
}

static __global__ void DataExtSetRodft01_1d(float *d_in, cufftComplex *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
	{
		d_out[ix+1].y = -d_in[ix];
		d_out[ix+1].x = 0.0f;
		if (ix<n-1) 
		{
			d_out[2*n-1-ix].y = d_in[ix];
			d_out[2*n-1-ix].x = 0.0f;
		}
	}
	d_out[0].x = d_out[0].y = 0.0f;
}

static __global__ void DataSubGetRodft01_1d(cufftComplex *d_in, float *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
		d_out[ix] = d_in[ix].x;
}

static __global__ void PhaseShiftForw_1d(cufftComplex *d_in, cufftComplex *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	float d_k;

	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
	{
		if (ix<n/2) 
			d_k = (float)ix*pi/(float)(n/2);
		else
			d_k = -pi+(float)(ix-n/2)*pi/(float)(n/2);

		d_out[ix].x = d_in[ix].x*cosf(d_k/2.0f)+d_in[ix].y*sinf(d_k/2.0f);
		d_out[ix].y = -d_in[ix].x*sinf(d_k/2.0f)+d_in[ix].y*cosf(d_k/2.0f);
	}
}

static __global__ void PhaseShiftBack_1d(cufftComplex *d_in, cufftComplex *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	float d_k;

	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
	{
		if (ix<n/2) 
			d_k = (float)ix*pi/(float)(n/2);
		else
			d_k = -pi+(float)(ix-n/2)*pi/(float)(n/2);

		d_out[ix].x = d_in[ix].x*cosf(d_k/2.0f)-d_in[ix].y*sinf(d_k/2.0f);
		d_out[ix].y = d_in[ix].x*sinf(d_k/2.0f)+d_in[ix].y*cosf(d_k/2.0f);
	}
}

void data_test_forw_1d(float *h_in, float *h_out, int n)
{
	float *d_in;
	cudaMalloc((void **)&d_in, n*sizeof(float));
	cudaMemcpy(d_in, h_in, n*sizeof(float), cudaMemcpyHostToDevice);
	
	float *d_out;
	cudaMalloc((void **)&d_out, n*sizeof(float));
	
	cufft_rodft00_1d(d_in, d_out, n);

	cudaMemcpy(h_out, d_out, n*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);
}

void data_test_back_1d(float *h_in, float *h_out, int n)
{
	float *d_in;
	cudaMalloc((void **)&d_in, n*sizeof(float));
	cudaMemcpy(d_in, h_in, n*sizeof(float), cudaMemcpyHostToDevice);
	
	float *d_out;
	cudaMalloc((void **)&d_out, n*sizeof(float));
	
	cufft_rodft01_1d(d_in, d_out, n);

	cudaMemcpy(h_out, d_out, n*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);
}

void data_test_derv_1d(float *h_in, float *h_out, float dx, int n)
{
	float *d_in;
	cudaMalloc((void **)&d_in, n*sizeof(float));
	cudaMemcpy(d_in, h_in, n*sizeof(float), cudaMemcpyHostToDevice);
	
	float *d_out;
	cudaMalloc((void **)&d_out, n*sizeof(float));
	
	cufft_dct2_idst1_1d(d_in, d_out, dx, n);

	cudaMemcpy(h_out, d_out, n*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);
}
