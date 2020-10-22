#ifndef CUFFT_DCT_DST_H
#define CUFFT_DCT_DST_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void cufft_dct2_idst1_1d(float *d_in, float *d_out, float dx, int nn);
void cufft_dct1_idst2_1d(float *d_in, float *d_out, float dx, int nn);
void cufft_dst2_idct1_1d(float *d_in, float *d_out, float dx, int nn);
void cufft_dst1_idct2_1d(float *d_in, float *d_out, float dx, int nn);

void data_test_derv_1d(float *h_in, float *h_out, float dx, int n);
void data_test_forw_1d(float *h_in, float *h_out, int n);
void data_test_back_1d(float *h_in, float *h_out, int n);
void data_test_derv_2d(float *h_in, float *h_out, float dx, int nx, int ny);
void data_test_dct_2d(float *h_in, float *h_out, int nx, int ny);

#endif
