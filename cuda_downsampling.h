// cuda_downsampling.h
#ifndef CUDA_DOWNSAMPLING_H
#define CUDA_DOWNSAMPLING_H

#include <stdio.h>
#include <stddef.h>
#include <iostream>
#include "ReadSrcFileHead.h"
#include <cufft.h>

struct FFTPlanCache {
    cufftHandle plan_forward;
    cufftHandle plan_inverse;
    int srcLen;
    int dstLen;
    int nchan;
    bool is_valid;
    
    cuFloatComplex* d_src;
    cuFloatComplex* d_fft;
    cuFloatComplex* d_trunc;
    cuFloatComplex* d_ifft;
    short* d_raw_input;
    cufftComplex* d_complex_output;
    
    cudaStream_t stream;
};

FFTPlanCache* create_fft_plan_cache(int nchan, int srcLen, int dstLen);
void destroy_fft_plan_cache(FFTPlanCache* cache);
float* batch_descend_sample_gpu_with_cache(const float* h_src, int nchan, int srcLen, int newLen, FFTPlanCache* cache);

#ifdef __cplusplus
extern "C" {
#endif

    short* cuda_downsample_data(FILE* fid, int nchan, int len,
        int targetFs, int originalFs,
        size_t* outputSize, int downsampleAllChannels);
        
    short* cuda_downsample_data_with_cache(FILE* fid, int nchan, int len,
        int targetFs, int originalFs,
        size_t* outputSize, int downsampleAllChannels, FFTPlanCache* cache);
        
    cufftComplex* cuda_downsample_data_complex_with_cache(FILE* fid, int nchan, int len,
        int targetFs, int originalFs,
        int* outputSamples, int downsampleAllChannels, FFTPlanCache* cache);

    cufftComplex* cuda_downsample_data_complex_dual_channel_with_cache(FILE* fid, int nchan, int len,
        int targetFs, int originalFs,
        int* outputSamples, int ref_channel, int monitor_channel, FFTPlanCache* cache);

    cufftComplex* cuda_downsample_data_complex_dual_channel_device_only_with_cache(
        FILE* fid, int nchan, int len,
        int targetFs, int originalFs,
        int* outputSamples, int ref_channel, int monitor_channel,
        FFTPlanCache* cache);
        
    float* read_data_cuda_three_channel(FILE* fid, int nchan, int len, int* success, int ch0, int ch1, int ch2);
    
    cufftComplex* cuda_downsample_data_complex_three_channel_device_only_with_cache(
        FILE* fid, int nchan, int len,
        int targetFs, int originalFs,
        int* outputSamples, int ch0, int ch1, int ch2,
        FFTPlanCache* cache);

    float* read_data_cuda(FILE* fid, int nchan, int len, int* success, int readAllChannels);
    
    float* read_data_cuda_dual_channel(FILE* fid, int nchan, int len, int* success, int ref_channel, int monitor_channel);
    
    float* read_data_cuda_gpu(FILE* fid, int nchan, int len, int* success, int readAllChannels);

    float* descend_sample_rate_gpu(const float* h_src, int srcLen, int newLen); 

    float* batch_descend_sample_gpu(const float* h_src, int nchan, int srcLen, int newLen);

#ifdef __cplusplus
}
#endif

#endif // CUDA_DOWNSAMPLING_H
