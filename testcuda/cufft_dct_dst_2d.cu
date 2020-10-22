#include "cufft_dct_dst.h"
#include "transpose_kernel.h"
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

#include "cufft.h"

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

#include "bmpio.h"
using std::ifstream;
using std::string;

using std::endl;
using std::ios;
//using std::cout;
//using std::setiosflags;
//using std::setprecision;

using namespace cv;

#define NORMAL_W 480
#define NORMAL_H 270

#ifndef threads_num
#define threads_num 256
#endif

#ifndef CV_PI
#define CV_PI   3.1415926535897932384626433832795
#endif

#ifndef CV_PI_F
#ifndef CV_PI
#define CV_PI_F 3.14159265f
#else
#define CV_PI_F ((float)CV_PI)
#endif
#endif

#ifndef CV_SIN45_F
#define CV_SIN45_F ((float)0.70710678118654752440084436210485)
#endif

typedef cuComplex cufftComplex;

void cufft_dct2_idst1_2d_x(float *d_in, float *d_out, float dx, int nx, int ny)
{
	for (int iy=0; iy<ny; iy++)
		cufft_dct2_idst1_1d(&d_in[iy*nx], &d_out[iy*nx], dx, nx);
}

void cufft_dst2_idct1_2d_x(float *d_in, float *d_out, float dx, int nx, int ny)
{
	for (int iy=0; iy<ny; iy++)
		cufft_dst2_idct1_1d(&d_in[iy*nx], &d_out[iy*nx], dx, nx);
}

void cufft_dct1_idst2_2d_x(float *d_in, float *d_out, float dx, int nx, int ny)
{
	for (int iy=0; iy<ny; iy++)
		cufft_dct1_idst2_1d(&d_in[iy*nx], &d_out[iy*nx], dx, nx);
}

void cufft_dst1_idct2_2d_x(float *d_in, float *d_out, float dx, int nx, int ny)
{
	for (int iy=0; iy<ny; iy++)
		cufft_dst1_idct2_1d(&d_in[iy*nx], &d_out[iy*nx], dx, nx);
}

void cufft_dct2_idst1_2d_y(float *d_in, float *d_out, float dy, int nx, int ny)
{
	/* Allocate memory for transpose matrix */
	float *d_in_trsp;
	cudaMalloc((void **)&d_in_trsp, nx*ny*sizeof(float));
	
	dim3 grids((nx+BLOCK_SIZE-1)/BLOCK_SIZE, (ny+BLOCK_SIZE-1)/BLOCK_SIZE), blocks(BLOCK_SIZE, BLOCK_SIZE);

	/* Transpose the matrix */
	transposeReal_2d<<<grids, blocks>>>(d_in_trsp, d_in, nx, ny);

	/* Allocate memory for derivative */
	float *d_in_tmp;
	cudaMalloc((void **)&d_in_tmp, nx*ny*sizeof(float));
	
	cufft_dct2_idst1_2d_x(d_in_trsp, d_in_tmp, dy, ny, nx);

	/* Transpose back the derivative */
	dim3 grids2((ny+BLOCK_SIZE-1)/BLOCK_SIZE, (nx+BLOCK_SIZE-1)/BLOCK_SIZE), blocks2(BLOCK_SIZE, BLOCK_SIZE);
	transposeReal_2d<<<grids2, blocks2>>>(d_out, d_in_tmp, ny, nx);

	/* Deallocate memory */
	cudaFree(d_in_trsp);
	cudaFree(d_in_tmp);
}

void cufft_dst2_idct1_2d_y(float *d_in, float *d_out, float dy, int nx, int ny)
{
	/* Allocate memory for transpose matrix */
	float *d_in_trsp;
	cudaMalloc((void **)&d_in_trsp, nx*ny*sizeof(float));
	
	dim3 grids((nx+BLOCK_SIZE-1)/BLOCK_SIZE, (ny+BLOCK_SIZE-1)/BLOCK_SIZE), blocks(BLOCK_SIZE, BLOCK_SIZE);

	/* Transpose the matrix */
	transposeReal_2d<<<grids, blocks>>>(d_in_trsp, d_in, nx, ny);

	/* Allocate memory for derivative */
	float *d_in_tmp;
	cudaMalloc((void **)&d_in_tmp, nx*ny*sizeof(float));
	
	cufft_dst2_idct1_2d_x(d_in_trsp, d_in_tmp, dy, ny, nx);

	/* Transpose back the derivative */
	dim3 grids2((ny+BLOCK_SIZE-1)/BLOCK_SIZE, (nx+BLOCK_SIZE-1)/BLOCK_SIZE), blocks2(BLOCK_SIZE, BLOCK_SIZE);
	transposeReal_2d<<<grids2, blocks2>>>(d_out, d_in_tmp, ny, nx);

	/* Deallocate memory */
	cudaFree(d_in_trsp);
	cudaFree(d_in_tmp);
}

void cufft_dct1_idst2_2d_y(float *d_in, float *d_out, float dy, int nx, int ny)
{
	/* Allocate memory for transpose matrix */
	float *d_in_trsp;
	cudaMalloc((void **)&d_in_trsp, nx*ny*sizeof(float));
	
	dim3 grids((nx+BLOCK_SIZE-1)/BLOCK_SIZE, (ny+BLOCK_SIZE-1)/BLOCK_SIZE), blocks(BLOCK_SIZE, BLOCK_SIZE);

	/* Transpose the matrix */
	transposeReal_2d<<<grids, blocks>>>(d_in_trsp, d_in, nx, ny);

	/* Allocate memory for derivative */
	float *d_in_tmp;
	cudaMalloc((void **)&d_in_tmp, nx*ny*sizeof(float));
	
	cufft_dct1_idst2_2d_x(d_in_trsp, d_in_tmp, dy, ny, nx);

	/* Transpose back the derivative */
	dim3 grids2((ny+BLOCK_SIZE-1)/BLOCK_SIZE, (nx+BLOCK_SIZE-1)/BLOCK_SIZE), blocks2(BLOCK_SIZE, BLOCK_SIZE);
	transposeReal_2d<<<grids2, blocks2>>>(d_out, d_in_tmp, ny, nx);

	/* Deallocate memory */
	cudaFree(d_in_trsp);
	cudaFree(d_in_tmp);
}

void cufft_dst1_idct2_2d_y(float *d_in, float *d_out, float dy, int nx, int ny)
{
	/* Allocate memory for transpose matrix */
	float *d_in_trsp;
	cudaMalloc((void **)&d_in_trsp, nx*ny*sizeof(float));
	
	dim3 grids((nx+BLOCK_SIZE-1)/BLOCK_SIZE, (ny+BLOCK_SIZE-1)/BLOCK_SIZE), blocks(BLOCK_SIZE, BLOCK_SIZE);

	/* Transpose the matrix */
	transposeReal_2d<<<grids, blocks>>>(d_in_trsp, d_in, nx, ny);

	/* Allocate memory for derivative */
	float *d_in_tmp;
	cudaMalloc((void **)&d_in_tmp, nx*ny*sizeof(float));
	
	cufft_dst1_idct2_2d_x(d_in_trsp, d_in_tmp, dy, ny, nx);

	/* Transpose back the derivative */
	dim3 grids2((ny+BLOCK_SIZE-1)/BLOCK_SIZE, (nx+BLOCK_SIZE-1)/BLOCK_SIZE), blocks2(BLOCK_SIZE, BLOCK_SIZE);
	transposeReal_2d<<<grids2, blocks2>>>(d_out, d_in_tmp, ny, nx);

	/* Deallocate memory */
	cudaFree(d_in_trsp);
	cudaFree(d_in_tmp);
}

/* test functions of the derivatives */
void data_test_derv_2d(float *h_in, float *h_out, float dx, int nx, int ny)
{
	float *d_in;
	cudaMalloc((void **)&d_in, nx*ny*sizeof(float));
	cudaMemcpy(d_in, h_in, nx*ny*sizeof(float), cudaMemcpyHostToDevice);
	
	float *d_out;
	cudaMalloc((void **)&d_out, nx*ny*sizeof(float));
	
	cufft_dct1_idst2_2d_y(d_in, d_out, dx, nx, ny);

	cudaMemcpy(h_out, d_out, nx*ny*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);
}

//#define pi 4.0f*atanf(1.0f)

static __global__ void R2CDataExt(float *d_in, cufftComplex *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
	{
		d_out[ix].x = d_in[ix];
		d_out[ix].y = 0.0f;
		d_out[2 * n - 1 - ix].x = d_in[ix];
		d_out[2 * n - 1 - ix].y = 0.0f;
	}
}
static __global__ void C2CHalf(cufftComplex *d_in, cufftComplex *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	float d_k;

	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
	{
		if (ix < n / 2)
			d_k = (float)ix*CV_PI_F / (float)(n / 2);
		else
			d_k = -CV_PI_F + (float)(ix - n / 2)*CV_PI_F / (float)(n / 2);

		d_out[ix].x = d_in[ix].x*cosf(d_k / 2.0f) + d_in[ix].y*sinf(d_k / 2.0f);
		d_out[ix].y = -d_in[ix].x*sinf(d_k / 2.0f) + d_in[ix].y*cosf(d_k / 2.0f);
	}
}
static __global__ void C2RDataGet(cufftComplex *d_in, float *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n) {
		//debug
		//if (ix < 40)
		//	d_out[ix] = 0.5f;
		//else
		d_out[ix] = d_in[ix].x;
	}
}

void cufft_DCT1D(float *d_in, float *d_out, int n)
{
	/* Allocate meory for extended complex data */
	cufftComplex *d_in_ext;
	cudaMalloc((void **)&d_in_ext, 2 * n * sizeof(cufftComplex));

	/* Extend and convert the data from float to cufftComplex */
	R2CDataExt << <(n + threads_num - 1) / threads_num, threads_num >> > (d_in, d_in_ext, n);

	/* Create a 1D FFT plan */
	cufftHandle plan;
	cufftPlan1d(&plan, 2 * n, CUFFT_C2C, 1);

	/* Allocate memory for transformed data */
	cufftComplex *d_out_ext;
	cudaMalloc((void **)&d_out_ext, 2 * n * sizeof(cufftComplex));

	/* Use the CUFFT plan to transform the signal out of place */
	cufftExecC2C(plan, d_in_ext, d_out_ext, CUFFT_FORWARD);

	/* Allocate memory for shifted data */
	cufftComplex *d_out_ext_shift;
	cudaMalloc((void **)&d_out_ext_shift, 2 * n * sizeof(cufftComplex));

	/* 1/2 phase shift */
	C2CHalf << <(2 * n + threads_num - 1) / threads_num, threads_num >> > (d_out_ext, d_out_ext_shift, 2 * n);

	/* Subtract the transformed data */
	C2RDataGet << <(n + threads_num - 1) / threads_num, threads_num >> > (d_out_ext_shift, d_out, n);

	cufftDestroy(plan);
	cudaFree(d_in_ext);
	cudaFree(d_out_ext);
	cudaFree(d_out_ext_shift);
}
void cufft_DCT_XY(float *d_in, float *d_out, int nx, int ny)
{
	for (int iy = 0; iy < ny; iy++) {
		cufft_DCT1D(&d_in[iy*nx], &d_out[iy*nx], nx);
		//d_out[iy*nx] = 1.0f;
	}
}
void cufft_DCT_2D(float *d_in, float *d_out, int nx, int ny)
{
	/* Allocate memory for derivative */
	float *d_in_tmp;
	cudaMalloc((void **)&d_in_tmp, nx*ny * sizeof(float));

	cufft_DCT_XY(d_in, d_in_tmp, nx, ny);
	/* Allocate memory for transpose matrix */
	float *d_in_trsp;
	cudaMalloc((void **)&d_in_trsp, nx*ny * sizeof(float));

	dim3 grids((nx + BLOCK_SIZE - 1) / BLOCK_SIZE, (ny + BLOCK_SIZE - 1) / BLOCK_SIZE), blocks(BLOCK_SIZE, BLOCK_SIZE);
	/* Transpose the matrix */
	transposeReal_2d << <grids, blocks >> > (d_in_trsp, d_in_tmp, nx, ny);

	cufft_DCT_XY(d_in_trsp, d_in_tmp, ny, nx);

	/* Transpose back the derivative */
	dim3 grids2((ny + BLOCK_SIZE - 1) / BLOCK_SIZE, (nx + BLOCK_SIZE - 1) / BLOCK_SIZE), blocks2(BLOCK_SIZE, BLOCK_SIZE);
	transposeReal_2d << <grids2, blocks2 >> > (d_out, d_in_tmp, ny, nx);

	/* Deallocate memory */
	cudaFree(d_in_trsp);
	cudaFree(d_in_tmp);
}
static __global__ void R2CDataExtN(float *d_in, cufftComplex *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
	{
		if (ix & 0x01) //odd
		{
			d_out[n - 1 - ix / 2].x = d_in[ix];
			d_out[n - 1 - ix / 2].y = 0.0f;
		}
		else //even
		{
			d_out[ix/2].x = d_in[ix];
			d_out[ix/2].y = 0.0f;
		}
	}
}

static __global__ void C2CHalfN(cufftComplex *d_in, cufftComplex *d_out, int n)
{
	//int tid = threadIdx.x;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	float d_k;

	//for (int ix=tid; ix<n; ix+=threads_num)
	if (ix < n)
	{	
		d_k = (float)ix*CV_PI_F / (float)(n * 2.0f);
		d_out[ix].x = (d_in[ix].x*cosf(d_k) - d_in[ix].y*sinf(d_k)) * sqrtf(2.0f / n);
		d_out[ix].y = (-d_in[ix].x*sinf(d_k) - d_in[ix].y*cosf(d_k)) * sqrtf(2.0f / n);
		if(ix==0)
			d_out[ix].x = d_out[ix].x * CV_SIN45_F;
	}
}
void cufft_DCT1DN(float *d_in, float *d_out, int n)
{
	/* Allocate meory for extended complex data */
	cufftComplex *d_in_ext;
	cudaMalloc((void **)&d_in_ext, n * sizeof(cufftComplex));

	/* Extend and convert the data from float to cufftComplex */
	R2CDataExtN << <(n + threads_num - 1) / threads_num, threads_num >> > (d_in, d_in_ext, n);

	/* Create a 1D FFT plan */
	cufftHandle plan;
	cufftPlan1d(&plan, n, CUFFT_C2C, 1);

	/* Allocate memory for transformed data */
	cufftComplex *d_out_ext;
	cudaMalloc((void **)&d_out_ext, n * sizeof(cufftComplex));

	/* Use the CUFFT plan to transform the signal out of place */
	cufftExecC2C(plan, d_in_ext, d_out_ext, CUFFT_FORWARD);

	/* Allocate memory for shifted data */
	cufftComplex *d_out_ext_shift;
	cudaMalloc((void **)&d_out_ext_shift, n * sizeof(cufftComplex));

	/* 1/2 phase shift */
	C2CHalfN << <( n + threads_num - 1) / threads_num, threads_num >> > (d_out_ext, d_out_ext_shift, n);

	/* Subtract the transformed data */
	C2RDataGet << <(n + threads_num - 1) / threads_num, threads_num >> > (d_out_ext_shift, d_out, n);

	cufftDestroy(plan);
	cudaFree(d_in_ext);
	cudaFree(d_out_ext);
	cudaFree(d_out_ext_shift);
}
void cufft_DCT_XYN(float *d_in, float *d_out, int nx, int ny)
{
	for (int iy = 0; iy < ny; iy++) {
		cufft_DCT1DN(&d_in[iy*nx], &d_out[iy*nx], nx);
		//d_out[iy*nx] = 1.0f;
	}
}
void cufft_DCT_2DN(float *d_in, float *d_out, int nx, int ny)
{
	/* Allocate memory for derivative */
	float *d_in_tmp;
	cudaMalloc((void **)&d_in_tmp, nx*ny * sizeof(float));

	cufft_DCT_XYN(d_in, d_in_tmp, nx, ny);
	/* Allocate memory for transpose matrix */
	float *d_in_trsp;
	cudaMalloc((void **)&d_in_trsp, nx*ny * sizeof(float));

	dim3 grids((nx + BLOCK_SIZE - 1) / BLOCK_SIZE, (ny + BLOCK_SIZE - 1) / BLOCK_SIZE), blocks(BLOCK_SIZE, BLOCK_SIZE);
	/* Transpose the matrix */
	transposeReal_2d << <grids, blocks >> > (d_in_trsp, d_in_tmp, nx, ny);

	cufft_DCT_XYN(d_in_trsp, d_in_tmp, ny, nx);

	/* Transpose back the derivative */
	dim3 grids2((ny + BLOCK_SIZE - 1) / BLOCK_SIZE, (nx + BLOCK_SIZE - 1) / BLOCK_SIZE), blocks2(BLOCK_SIZE, BLOCK_SIZE);
	transposeReal_2d << <grids2, blocks2 >> > (d_out, d_in_tmp, ny, nx);

	/* Deallocate memory */
	cudaFree(d_in_trsp);
	cudaFree(d_in_tmp);
}
void data_test_dct_2d(float *h_in, float *h_out, int nx, int ny)
{
	float *d_in;
	cudaMalloc((void **)&d_in, nx*ny * sizeof(float));
	cudaMemcpy(d_in, h_in, nx*ny * sizeof(float), cudaMemcpyHostToDevice);

	float *d_out;
	cudaMalloc((void **)&d_out, nx*ny * sizeof(float));

	//cufft_DCT_2D(d_in, d_out, nx, ny);
	cufft_DCT_2DN(d_in, d_out, nx, ny);

	cudaMemcpy(h_out, d_out, nx*ny * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);
}
int main()
{
#if 1 //Opecv buffer test

	clock_t star2, end2;
	float time2 = 0;
	Mat reference_frame, rendition_frame, next_reference_frame, next_rendition_frame;
	Mat reference_frame_v, rendition_frame_v, next_reference_frame_v, next_rendition_frame_v;
	Mat reference_frame_float, rendition_frame_float, reference_dct, rendition_dct;
	cudaEvent_t start2, stop2;
	float* pdctbuff;	
	float dx = 0.5;

	reference_frame = imread("d:/tmp/bmptest/reference_frame.bmp");
	rendition_frame = imread("d:/tmp/bmptest/rendition_frame.bmp");
	next_reference_frame = imread("d:/tmp/bmptest/next_reference_frame.bmp");
	next_rendition_frame = imread("d:/tmp/bmptest/next_rendition_frame.bmp");

	cvtColor(reference_frame, reference_frame_v, COLOR_BGR2HSV);
	cvtColor(rendition_frame, rendition_frame_v, COLOR_BGR2HSV);
	cvtColor(next_reference_frame, next_reference_frame_v, COLOR_BGR2HSV);
	//cvtColor(next_rendition_frame, next_rendition_frame_v, COLOR_BGR2HSV);

	extractChannel(reference_frame_v, reference_frame_v, 2);
	extractChannel(rendition_frame_v, rendition_frame_v, 2);
	extractChannel(next_reference_frame_v, next_reference_frame_v, 2);
	//extractChannel(next_rendition_frame_v, next_rendition_frame_v, 2);

	reference_frame_v.convertTo(reference_frame_v, CV_32FC1, 1.0 / 255.0);
	rendition_frame_v.convertTo(rendition_frame_v, CV_32FC1, 1.0 / 255.0);

	//OpenCV DCT 
	dct(reference_frame_v, reference_dct);
	
	//Cufft DCT algorithm 
	pdctbuff = (float*)malloc(reference_frame_v.cols*reference_frame_v.rows * sizeof(float));
	data_test_dct_2d((float*)reference_frame_v.data, pdctbuff, reference_frame_v.cols, reference_frame_v.rows);
	WriteFloatBmp("d:/tmp/bmptest/2dfftdct_frame.bmp", NORMAL_W, NORMAL_H, (float*)pdctbuff);
	//free DCT buffer
	if (pdctbuff)
		free(pdctbuff);	
#else

	int nx = NORMAL_W, ny = NORMAL_H;
	float dx = 1.0;
	float *h_in = (float *)malloc(nx*ny * sizeof(float));
	float *h_syn = (float *)malloc(nx*ny * sizeof(float));
	for (int ix = 0; ix < nx; ix++)
		for (int iy = 0; iy < ny; iy++)
		{
			h_in[iy*nx + ix] = expf(-powf((iy - ny / 2)*dx, 2.0f));
			h_syn[iy*nx + ix] = -2.0f*(iy - ny / 2 + 0.5f)*dx*expf(-powf((iy - ny / 2 + 0.5f)*dx, 2.0f));
		}

	float *h_out = (float *)malloc(nx*ny * sizeof(float));
	data_test_derv_2d(h_in, h_out, dx, nx, ny);

	//	for (int iy=0; iy<ny; iy++)
	//for (int iy = 0; iy < ny; iy++)
	//	printf("%d %f %f\n", iy, h_out[iy*nx], h_syn[iy*nx]);

	WriteFloatBmp("d:/tmp/bmptest/2dfftdct_frame.bmp", NORMAL_W, NORMAL_H, (float*)h_out);


	free(h_in);
	free(h_out);
	free(h_syn);
#endif
	return EXIT_SUCCESS;
}
