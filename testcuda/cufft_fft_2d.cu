#include "cufft_fft.h"
#include "transpose_kernel.h"
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

void cufft_fftf_2d_x(float *d_in, float *d_out, float dx, int nx, int ny)
{
	for (int iy=0; iy<ny; iy++)
		cufft_fftf_1d(&d_in[iy*nx], &d_out[iy*nx], dx, nx);
}

void cufft_fftb_2d_x(float *d_in, float *d_out, float dx, int nx, int ny)
{
	for (int iy=0; iy<ny; iy++)
		cufft_fftf_1d(&d_in[iy*nx], &d_out[iy*nx], dx, nx);
}

void cufft_fftf_2d_y(float *d_in, float *d_out, float dy, int nx, int ny)
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
	
	cufft_fftf_2d_x(d_in_trsp, d_in_tmp, dy, ny, nx);

	/* Transpose back the derivative */
	dim3 grids2((ny+BLOCK_SIZE-1)/BLOCK_SIZE, (nx+BLOCK_SIZE-1)/BLOCK_SIZE), blocks2(BLOCK_SIZE, BLOCK_SIZE);
	transposeReal_2d<<<grids2, blocks2>>>(d_out, d_in_tmp, ny, nx);

	/* Deallocate memory */
	cudaFree(d_in_trsp);
	cudaFree(d_in_tmp);

}

void cufft_fftb_2d_y(float *d_in, float *d_out, float dy, int nx, int ny)
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
	
	cufft_fftb_2d_x(d_in_trsp, d_in_tmp, dy, ny, nx);

	/* Transpose back the derivative */
	dim3 grids2((ny+BLOCK_SIZE-1)/BLOCK_SIZE, (nx+BLOCK_SIZE-1)/BLOCK_SIZE), blocks2(BLOCK_SIZE, BLOCK_SIZE);
	transposeReal_2d<<<grids2, blocks2>>>(d_out, d_in_tmp, ny, nx);

	/* Deallocate memory */
	cudaFree(d_in_trsp);
	cudaFree(d_in_tmp);
}

