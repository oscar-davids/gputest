#include "cufft_dct_dst.h"

#define pi 4.0f*atanf(1.0f)

int main()
{
	int nx = 120, ny=408;
	float dx = 0.1;
	float *h_in  = (float *)malloc(nx*ny*sizeof(float));
	float *h_syn = (float *)malloc(nx*ny*sizeof(float));
	for (int ix=0; ix<nx; ix++)
	for (int iy=0; iy<ny; iy++)
	{
		h_in[iy*nx+ix] = expf(-powf((iy-ny/2)*dx, 2.0f));
		h_syn[iy*nx+ix] = -2.0f*(iy-ny/2+0.5f)*dx*expf(-powf((iy-ny/2+0.5f)*dx, 2.0f));
	}
	
	float *h_out = (float *)malloc(nx*ny*sizeof(float));
	data_test_derv_2d(h_in, h_out, dx, nx, ny);

//	for (int iy=0; iy<ny; iy++)
	for (int iy=0; iy<ny; iy++) 
		printf("%d %f %f\n", iy, h_out[iy*nx], h_syn[iy*nx]);

	free(h_in);
	free(h_out);
	free(h_syn);
	return EXIT_SUCCESS;
}
