#include "cufft_dct_dst.h"
#include "transpose_kernel.h"
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

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

#define pi 4.0f*atanf(1.0f)

int main()
{
	int nx = 120, ny = 408;
	float dx = 0.5;
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
	for (int iy = 0; iy < ny; iy++)
		printf("%d %f %f\n", iy, h_out[iy*nx], h_syn[iy*nx]);

	free(h_in);
	free(h_out);
	free(h_syn);
	return EXIT_SUCCESS;
}
