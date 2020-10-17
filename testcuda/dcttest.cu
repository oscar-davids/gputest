//#include<iostream>
//#include<fstream>
//#include<math.h>
//#include<string>
//#include<stdio.h>
//#include<cuda_runtime.h>
//#include<device_launch_parameters.h>
//using namespace std;

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

#include <cmath>
#include <cstdio>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

#include "bmpio.h"
//using namespace std;
using std::ifstream;
using std::string;
using std::cout;
using std::endl;
using std::ios;
using std::setiosflags;
using std::setprecision;

using namespace cv;

//#define length 8
#define PI 3.14159265
#define length 256
#define block_len 16

#define IMG_WIDTH	480
#define IMG_HEIGHT	270


cudaError_t dctWithCuda_1(const float *d, float *D);

cudaError_t dctWithCuda_2(const float *f, float *F, int nwidth, int nheight);

void dct(float *f, float *F){
	int i,j,t;
	//float data[length]={0.0};
	float tmp;

	float data[length] = {0.0};
	for(t=0; t<length; t++)
	{
		for (i=0; i<length; i++)
				data[i] = f[t*length+i];//load row data from f.

		for(i=0; i<length; i++)
		{
			if(i==0)
			{
				tmp = (float)(1.0/sqrt(1.0*length));
				F[t*length+i] = 0.0;//why use F[bid]? Do transpose at the same time.
				for(j=0; j<length; j++)
					F[t*length+i] +=data[j] ;
				F[t*length] *= tmp;
			}
			else
			{
				tmp = (float)(sqrt(2.0/(1.0*length)));
				for(i=1; i<length; i++){
					F[t*length+i] = 0;
					for(j=0; j<length; j++)
						F[t*length+i] += (float)(data[j]*cos((2*j+1)*i*PI/(2.0*length)));
					F[t*length+i] *= tmp;
				}
			}
		}
	}

	for(t=0; t<length; t++)
	{
		for(i=0; i<length; i++)
			data[i] = F[i*length+t];
		for(i=0; i<length; i++)
		{
			if(i==0)
			{
				tmp=(float)(1.0/sqrt(1.0*length));
				F[t]=0;
				for(j=0; j<length; j++)
					F[t] += data[j];
				F[t] *= tmp;
			}
			else
			{
				tmp = (float)(sqrt(2.0/(1.0*length)));
				for(i=1; i<length; i++)
				{
					F[i*length+t] = 0;
					for(j=0; j<length; j++)
						F[i*length+t] += (float)(data[j]*cos((2*j+1)*i*PI/(2*length)));
					F[i*length+t] *= tmp;
				}
			}
		}
	}
}

__global__ void dct_1(const float *f,float *F){
	int bid = blockIdx.x;
	//int tid = threadIdx.x;
	int i,j;
	//float data[length]={0.0};
	float tmp;
	//printf("");
	if(bid<length){
		float data[length];
		for (i=0; i<length; i++)
			data[i] = f[bid*length+i];//load row data from f.
		__syncthreads();
		for(i=0; i<length; i++){
			if(i==0){
				tmp = (float)(1.0/sqrt(1.0*length));
				F[bid * length + i] = 0.0;
				for(j=0; j<length; j++)
					F[bid*length+i] +=data[j] ;
				F[bid*length] *= tmp;
			}
			else{
				tmp = (float)(sqrt(2.0/(1.0*length)));
				for(i=1; i<length; i++){
					F[bid*length+i] = 0;
					for(j=0; j<length; j++)
						F[bid*length+i] += (float)(data[j]*cos((2*j+1)*i*PI/(2*length)));
					F[bid*length+i] *= tmp;
				}
			}
		}
		__syncthreads();
		for(i=0; i<length; i++)
			data[i] = F[i*length+bid];
		__syncthreads();
		for(i=0; i<length; i++){
			if(i==0){
				tmp=(float)(1.0/sqrt(1.0*length));
				F[bid]=0;
				for(j=0; j<length; j++)
					F[bid] += data[j];
				F[bid] *= tmp;
			}
			else{
				tmp = (float)(sqrt(2.0/(1.0*length)));
				for(i=1; i<length; i++){
					F[i*length+bid] = 0;
					for(j=0; j<length; j++)
						F[i*length+bid] += (float)(data[j]*cos((2*j+1)*i*PI/(2*length)));
					F[i*length+bid] *= tmp;
				}
			}
		}
		__syncthreads();
	}
}

__global__ void dct_2(const float *f, float *F, int nwidth, int nheight){
	int tidx = blockIdx.x*blockDim.x + threadIdx.x;
	int tidy = blockIdx.y*blockDim.y + threadIdx.y;
	int index = tidy * nwidth + tidx;
	int i;
	float tmp;
	float beta ,alfa;
	if(tidx == 0)
		beta = sqrt(1.0/ nwidth);
	else
		beta = sqrt(2.0/ nwidth);

	if(tidy == 0)
		alfa = sqrt(1.0/ nheight);
	else
		alfa = sqrt(2.0/ nheight);

	if(tidx< nwidth && tidy< nheight)
	{
		for(i=0; i< nwidth * nheight; i++)
		{
			int x = i % nwidth;
			int y = i / nwidth;
			tmp += f[i]*::cos((2*x+1)*tidx*PI/(2.0*nwidth))*
				::cos((2*y+1)*tidy*PI/(2.0*nheight));
		}
		F[index]=(float)alfa * beta * tmp;
	}
}

int main(){
	ifstream infile("gradient.txt");
	int i=0;
	string line;
	//float f[length*length] = {0,0};
	//float F0[length*length] = {0.0};
	//float F1[length*length] = {0.0};
	//float F2[length*length] = {0.0};

	float *f = (float*)malloc(length*length * sizeof(float));
	float *F0 = (float*)malloc(length*length * sizeof(float));
	float *F1 = (float*)malloc(length*length * sizeof(float));
	float *F2 = (float*)malloc(IMG_WIDTH *IMG_HEIGHT * sizeof(float));
	

	while(i<length*length){
		if(getline(infile, line))
		{
			f[i] = atof(line.c_str());
		}
		i++;
	}

	Mat reference_frame, rendition_frame, next_reference_frame, next_rendition_frame;
	Mat reference_frame_v, rendition_frame_v, next_reference_frame_v, next_rendition_frame_v;
	Mat reference_frame_float, rendition_frame_float, reference_dct, rendition_dct;

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

//	cout<<"before"<<endl;
//	for(i=0; i<length*length; i++){
//			cout<<f[i]<<" ";
//			if ((i+1)%length==0)
//				cout<<endl;
//		}
//	cout<<endl;
//	for(i=0; i<length*length; i++){
//			cout<<F1[i]<<" ";
//			if ((i+1)%length==0)
//					cout<<endl;
//	}

	clock_t star0, end0;
	clock_t star1, end1;
	clock_t star2, end2;

	//use event to record time
	//float time0 = 0;
	float time1 = 0;
	float time2 = 0;
//	cudaEvent_t start0, stop0;
//	cudaEventCreate(&start0);
//	cudaEventCreate(&stop0);

	cudaEvent_t start1, stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);

	cudaEvent_t start2, stop2;
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);

	/*
	 * excute dct()
	 */
	star0 = clock();
	dct(f,F0);
	end0 = clock();

//	cout<<"----------------dct()-----------"<<endl;
//	for (i = 0; i < length * length; i++)
//	{
//		 /* GPU can't calculate floating number precisely
//		 * 0 will be a very small floating number.
//		 * so print this numbers with 7 digits after decimal point
//		*/
//		cout << setiosflags(ios::right)
//			<< setiosflags(ios::fixed) << setprecision(7)
//			<< F0[i] << "\t";
//		if ((i + 1) % length == 0)
//			cout << endl;
//	}

#if 0
	/*
	 * excute dct_1()
	 */
	star1 = clock();
	cudaEventRecord(start1, 0 );
	cudaError_t cudaStatus = dctWithCuda_1(f,F1);
	if (cudaStatus != cudaSuccess)
	{
	        fprintf(stderr, "dctWithCuda_1 failed!");
	        return 1;
	}
	end1 = clock();
	cudaEventRecord(stop1, 0 );

	cudaEventSynchronize(start1);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&time1,start1,stop1);

	printf("excute1 time: %f (ms)\n",time1);
	cudaEventDestroy(start1);    //destory the event
	cudaEventDestroy(stop1);
#endif

//	cout<<"----------------dct_1()-----------"<<endl;
//	for (i = 0; i < length * length; i++)
//	{
//		 /* GPU can't calculate floating number precisely
//		 * 0 will be a very small floating number.
//		 * so print this numbers with 7 digits after decimal point
//		*/
//		cout << setiosflags(ios::right)
//			<< setiosflags(ios::fixed) << setprecision(7)
//			<< F1[i] << "\t";
//		if ((i + 1) % length == 0)
//			cout << endl;
//	}


	/*
	 * excute dct_2()
	 */
	star2 = clock();
	cudaEventRecord(start2, 0 );
	cudaError_t cudaStatus_ = dctWithCuda_2((float*)reference_frame_v.data,F2, IMG_WIDTH, IMG_HEIGHT);
	if (cudaStatus_ != cudaSuccess)
	{
			fprintf(stderr, "dctWithCuda_1 failed!");
		    return 1;
	}
	cudaEventRecord(stop2, 0 );
	end2 = clock();

	cudaEventSynchronize(start2);
	cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&time2,start2,stop2);

	printf("excute2 time: %f (ms)\n",time2);
	cudaEventDestroy(start2);    //destory the event
	cudaEventDestroy(stop2);

	WriteFloatBmp("d:/tmp/bmptest/2ddct_frame.bmp", 480, 270, (float*)F2);


	cv::gpu::GpuMat gmatreference_frame_v, gmatreference_dct, gmatrendition_dct, gmatdiff_dct;

	gmatreference_frame_v.upload(reference_frame_v);
	cv::gpu::dct2d(gmatreference_frame_v, gmatreference_dct);
	Mat tmp_frame;
	gmatreference_dct.download(tmp_frame);
	WriteFloatBmp("d:/tmp/bmptest/2gpu_dct_reference_frame.bmp", 480, 270, (float*)tmp_frame.data);


//	for (i = 0; i < length * length; i++)
//	{
//		 /* GPU can't calculate floating number precisely
//		 * 0 will be a very small floating number.
//		 * so print this numbers with 7 digits after decimal point
//		*/
//		cout << setiosflags(ios::right)
//			<< setiosflags(ios::fixed) << setprecision(7)
//			<< F2[i] << "\t";
//		if ((i + 1) % length == 0)
//			cout << endl;
//	}


	//time
	cout<<"----------------clock()-----------"<<endl;
	cout<< "dct() timeused="<<end0-star0<<"ms"<<endl;
	cout<< "dct_1() timeused="<<end1-star1<<"ms"<<endl;
	cout<< "dct_2() timeused="<<end2-star2<<"ms"<<endl;

//	cout<<"after"<<endl;
//	for(i=0; i<length*length; i++){
//		cout<<f[i]<<" ";
//		if ((i+1)%length==0)
//			cout<<endl;
//	}
//	cout<<endl;
//	for(i=0; i<length*length; i++){
//			cout<<F[i]<<" ";
//			if ((i+1)%length==0)
//					cout<<endl;
//	}

	return 0;

}

cudaError_t dctWithCuda_1(const float *d, float *D){
	float *dev_d = 0;
	float *dev_D = 0;
	float time=0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_d,length *length* sizeof(float));
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_D,length *length* sizeof(float));
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_d, d,length *length*sizeof(float),cudaMemcpyHostToDevice);
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMemcpy-- failed");
		goto Error;
	}
	//launch a kernel on the GPU
	dct_1<<<length,1>>>(dev_d, dev_D);

	cudaStatus = cudaThreadSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaThreadSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0 );
    cudaStatus = cudaMemcpy(D, dev_D, length*length* sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
    cudaEventRecord(stop, 0 );
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time,start,stop);

    printf("copy1 time: %f (ms)\n",time);

Error:
	cudaFree(dev_d);
	cudaFree(dev_D);
	cudaEventDestroy(start);    //destory the event
	cudaEventDestroy(stop);
	return cudaStatus;
}


cudaError_t dctWithCuda_2(const float *d, float *D, int nwidth, int nheight){
	float *dev_d = 0;
	float *dev_D = 0;
	float time=0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_d, nwidth * nheight * sizeof(float));
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_D, nwidth * nheight * sizeof(float));
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_d, d, nwidth * nheight * sizeof(float),cudaMemcpyHostToDevice);
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMalloc failed");
		goto Error;
	}

	dim3 grid(1, 1, 1);
	dim3 dimblock(16, 16);	

	grid.x = (nwidth + 15) / 16;
	grid.y = (nheight + 15) / 16;

	dct_2 <<<grid, dimblock >>> (dev_d, dev_D, nwidth, nheight);
	//launch a kernel on the GPU
	//dct_2<<<1, (nwidth /block_len)*(nheight /block_len), block_len*block_len>>>(dev_d, dev_D, nwidth, nheight);

	cudaStatus = cudaThreadSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaThreadSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0 );
    cudaStatus = cudaMemcpy(D, dev_D, nwidth*nheight * sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
    cudaEventRecord(stop, 0 );
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time,start,stop);

    printf("copy2 time: %f (ms)\n",time);
Error:
	cudaFree(dev_d);
	cudaFree(dev_D);
	cudaEventDestroy(start);    //destory the event
	cudaEventDestroy(stop);
	return cudaStatus;
}















