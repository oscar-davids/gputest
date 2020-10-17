#include "cufft.h"
#ifndef pi
#define pi 4.0f*atanf(1.0f)
#endif
#ifndef threads_num
#define threads_num 256
#endif

static __global__ void DataTypeConvertFloatToComplex_1d(float *d_in, cufftComplex *d_out, int n);
static __global__ void DataGetBackFft_1d(cufftComplex *d_in, float *d_out, float dx, int n);
static __global__ void PhaseShiftForwFft_1d(cufftComplex *d_in, cufftComplex *d_out, int n);
static __global__ void PhaseShiftBackFft_1d(cufftComplex *d_in, cufftComplex *d_out, int n);

void cufft_fftf_1d(float *d_in, float *d_out, float dx, int n)
{
	/* Data conversion from float to Complex */
	cufftComplex *d_in_data;
	cudaMalloc((void **)&d_in_data, n*sizeof(cufftComplex));
	DataTypeConvertFloatToComplex_1d<<<(n+threads_num-1)/threads_num, threads_num>>>(d_in, d_in_data, n);

	/* Create a 1D forward/inverse FFT plan */
	cufftHandle plan;
	cufftPlan1d(&plan, n, CUFFT_C2C, 1);

	/* Use the CUFFT plan to transform the array */
	cufftComplex *d_tmp_data;
	cudaMalloc((void **)&d_tmp_data, n*sizeof(cufftComplex));
	cufftExecC2C(plan, d_in_data, d_tmp_data, CUFFT_FORWARD);

	/* 1/2 phase shift for staggered grid */
	cufftComplex *d_tmp_data_shift;
	cudaMalloc((void **)&d_tmp_data_shift, n*sizeof(cufftComplex));
	PhaseShiftForwFft_1d<<<(n+threads_num-1)/threads_num, threads_num>>>(d_tmp_data, d_tmp_data_shift, n);

	/* Inverse fourier transform */
	cufftComplex *d_out_data;
	cudaMalloc((void **)&d_out_data, n*sizeof(cufftComplex));
	cufftExecC2C(plan, d_tmp_data_shift, d_out_data, CUFFT_INVERSE);

	/* Data normalization*/
	DataGetBackFft_1d<<<(n+threads_num-1)/threads_num, threads_num>>>(d_out_data, d_out, dx, n);

	cufftDestroy(plan);
	cudaFree(d_in_data);
	cudaFree(d_tmp_data);
	cudaFree(d_tmp_data_shift);
	cudaFree(d_out_data);
}

void cufft_fftb_1d(float *d_in, float *d_out, float dx, int n)
{
	/* Data conversion from float to Complex */
	cufftComplex *d_in_data;
	cudaMalloc((void **)&d_in_data, n*sizeof(cufftComplex));
	DataTypeConvertFloatToComplex_1d<<<(n+threads_num-1)/threads_num, threads_num>>>(d_in, d_in_data, n);

	/* Create a 1D forward/inverse FFT plan */
	cufftHandle plan;
	cufftPlan1d(&plan, n, CUFFT_C2C, 1);

	/* Use the CUFFT plan to transform the array */
	cufftComplex *d_tmp_data;
	cudaMalloc((void **)&d_tmp_data, n*sizeof(cufftComplex));
	cufftExecC2C(plan, d_in_data, d_tmp_data, CUFFT_FORWARD);

	/* -1/2 phase shift for staggered grid */
	cufftComplex *d_tmp_data_shift;
	cudaMalloc((void **)&d_tmp_data_shift, n*sizeof(cufftComplex));
	PhaseShiftBackFft_1d<<<(n+threads_num-1)/threads_num, threads_num>>>(d_tmp_data, d_tmp_data_shift, n);

	/* Inverse fourier transform */
	cufftComplex *d_out_data;
	cudaMalloc((void **)&d_out_data, n*sizeof(cufftComplex));
	cufftExecC2C(plan, d_tmp_data_shift, d_out_data, CUFFT_INVERSE);

	/* Data normalization*/
	DataGetBackFft_1d<<<(n+threads_num-1)/threads_num, threads_num>>>(d_out_data, d_out, dx, n);

	cufftDestroy(plan);
	cudaFree(d_in_data);
	cudaFree(d_tmp_data);
	cudaFree(d_tmp_data_shift);
	cudaFree(d_out_data);
}

static __global__ void DataTypeConvertFloatToComplex_1d(float *d_in, cufftComplex *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
	{
		d_out[ix].x = d_in[ix];
		d_out[ix].y = 0.0f;
	}
}

static __global__ void DataGetBackFft_1d(cufftComplex *d_in, float *d_out, float dx, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
		d_out[ix] = d_in[ix].x/((float)n*dx);
}

static __global__ void PhaseShiftForwFft_1d(cufftComplex *d_in, cufftComplex *d_out, int n)
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
		
		d_out[ix].y = d_k*(d_in[ix].x*cosf(d_k/2.0f)+d_in[ix].y*sinf(d_k/2.0f));
		d_out[ix].x = d_k*(-d_in[ix].x*sinf(d_k/2.0f)+d_in[ix].y*cosf(d_k/2.0f));
	}
}

static __global__ void PhaseShiftBackFft_1d(cufftComplex *d_in, cufftComplex *d_out, int n)
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
		
		d_out[ix].y = d_k*(d_in[ix].x*cosf(d_k/2.0f)-d_in[ix].y*sinf(d_k/2.0f));
		d_out[ix].x = d_k*(d_in[ix].x*sinf(d_k/2.0f)+d_in[ix].y*cosf(d_k/2.0f));
	}
}

void data_fft_derv(float *h_in, float *h_out, float dx, int n)
{
	float *d_in;
	cudaMalloc((void **)&d_in, n*sizeof(float));
	cudaMemcpy(d_in, h_in, n*sizeof(float), cudaMemcpyHostToDevice);
	
	float *d_out;
	cudaMalloc((void **)&d_out, n*sizeof(float));
	
	cufft_fftb_1d(d_in, d_out, dx, n);

	cudaMemcpy(h_out, d_out, n*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);
}
