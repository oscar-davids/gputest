#ifndef CUFFT_FFT_H
#define CUFFT_FFT_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void cufft_fftf_1d(float *d_in, float *d_out, float dx, int n);
void cufft_fftb_1d(float *d_in, float *d_out, float dx, int n);
void data_fft_derv(float *h_in, float *h_out, float dx, int n);

#endif
