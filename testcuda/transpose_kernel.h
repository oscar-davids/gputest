#ifndef TRANSPOSE_KERNEL_H
#define TRANSPOSE_KERNEL_H

#define BLOCK_SIZE 16

/*static __global__ void transposeReal_2d(float *A_T, const float *A, int width, int height)
{
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < height && col < width)
		As[threadIdx.y][threadIdx.x] = A[row * width + col];
	__syncthreads();
	
	row = blockIdx.x * blockDim.x + threadIdx.y;
	col = blockIdx.y * blockDim.y + threadIdx.x;
	if (row < width && col < height)
		A_T[row * height + col] = As[threadIdx.x][threadIdx.y];
}
*/
static __global__ void transposeReal_2d(float *out, float *in, int width, int height)
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iy = blockIdx.y*blockDim.y+threadIdx.y;
	int idx_in 	= iy*width+ix;
	int idx_out = ix*height+iy;

	if (ix<width && iy<height)
		out[idx_out] = in[idx_in];
}

#endif
