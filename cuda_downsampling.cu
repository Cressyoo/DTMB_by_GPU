// cuda_downsampling.cu
#include "cuda_downsampling.h"
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
#include <iostream>
#include <chrono>
#include <string>
#include "ReadSrcFileHead.h"
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            fprintf(stderr, "CUFFT error in %s at line %d: code %d\n", \
                    __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void buildComplexFromInt16(const short* d_int16, cuFloatComplex* d_complex,
    int nchan, int len);
__global__ void packFrequencyDomain(const cuFloatComplex* d_fftSrc, cuFloatComplex* d_fftDst,
    int srcLen, int newLen);
__global__ void normalizeIFFT(cuFloatComplex* d_data, int len, float normFactor);
__global__ void complexToLinearInt16(const cuFloatComplex* d_complex, short* d_int16,
    int nchan, int newLen);

__global__ void buildComplexFromInt16(const short* d_int16, cuFloatComplex* d_complex,
    int nchan, int len) {
    int totalSamples = nchan * len;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSamples) return;

    int int16Idx = idx * 2;
    float real = (float)d_int16[int16Idx];
    float imag = (float)d_int16[int16Idx + 1];
    d_complex[idx] = make_cuFloatComplex(real, imag);
}

__global__ void packFrequencyDomain(const cuFloatComplex* d_fftSrc, cuFloatComplex* d_fftDst,
    int srcLen, int newLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= newLen) return;

    int halfNew = newLen / 2;
    if (idx <= halfNew) {
        d_fftDst[idx] = d_fftSrc[idx];
    }
    else {
        int srcIdx = srcLen - (newLen - idx);
        d_fftDst[idx] = d_fftSrc[srcIdx];
    }
}

__global__ void normalizeIFFT(cuFloatComplex* d_data, int len, float normFactor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;

    cuFloatComplex val = d_data[idx];
    d_data[idx] = make_cuFloatComplex(cuCrealf(val) * normFactor,
        -cuCimagf(val) * normFactor);
}

__global__ void complexToLinearInt16(const cuFloatComplex* d_complex, short* d_int16,
    int nchan, int newLen) {
    int totalSamples = nchan * newLen;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSamples) return;

    cuFloatComplex val = d_complex[idx];
    float real = cuCrealf(val);
    float imag = cuCimagf(val);

    int outIdx = idx * 2;
    d_int16[outIdx] = (short)real;
    d_int16[outIdx + 1] = (short)imag;
}

__global__ void convert_int16_to_complex_kernel(const short* d_buff, cufftComplex* d_complex,
    int nchan, int len, int outputChannels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalOutput = outputChannels * len;
    
    if (idx < totalOutput) {
        int ch = idx / len;
        int s = idx % len;
        int buffIdx = s * (2 * nchan) + 2 * ch;
        
        short real = d_buff[buffIdx];
        short imag = d_buff[buffIdx + 1];
        
        d_complex[idx].x = (float)real;
        d_complex[idx].y = (float)imag;
    }
}

float* read_data_cuda_gpu(FILE* fid, int nchan, int len, int* success, int readAllChannels) {
    int totalInt16 = 2 * nchan * len;
    short* h_buff = (short*)malloc(totalInt16 * sizeof(short));
    if (!h_buff) {
        *success = 0;
        return NULL;
    }

    size_t itemsRead = fread(h_buff, sizeof(short), totalInt16, fid);
    if (itemsRead < totalInt16) {
        free(h_buff);
        *success = 0;
        return NULL;
    }

    int outputChannels = readAllChannels ? nchan : 1;
    
    short* d_buff = NULL;
    cufftComplex* d_complex = NULL;
    float2* h_complexData = (float2*)malloc(outputChannels * len * sizeof(float2));
    
    if (!h_complexData) {
        free(h_buff);
        *success = 0;
        return NULL;
    }

    CUDA_CHECK(cudaMalloc(&d_buff, totalInt16 * sizeof(short)));
    CUDA_CHECK(cudaMalloc(&d_complex, outputChannels * len * sizeof(cufftComplex)));
    
    CUDA_CHECK(cudaMemcpy(d_buff, h_buff, totalInt16 * sizeof(short), cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int gridSize = (outputChannels * len + blockSize - 1) / blockSize;
    convert_int16_to_complex_kernel<<<gridSize, blockSize>>>(d_buff, d_complex, nchan, len, outputChannels);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(h_complexData, d_complex, outputChannels * len * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_buff));
    CUDA_CHECK(cudaFree(d_complex));
    free(h_buff);
    
    *success = 1;
    return (float*)h_complexData;
}

float* read_data_cuda(FILE* fid, int nchan, int len, int* success, int readAllChannels) {
    int totalInt16 = 2 * nchan * len;
    short* buff = (short*)malloc(totalInt16 * sizeof(short));
    if (!buff) {
        *success = 0;
        return NULL;
    }

    size_t itemsRead = fread(buff, sizeof(short), totalInt16, fid);
    if (itemsRead < totalInt16) {
        free(buff);
        *success = 0;
        return NULL;
    }

    int outputChannels = readAllChannels ? nchan : 1;
    float2* complexData = (float2*)malloc(outputChannels * len * sizeof(float2));
    if (!complexData) {
        free(buff);
        *success = 0;
        return NULL;
    }

    for (int ch = 0; ch < outputChannels; ch++) {
        for (int s = 0; s < len; s++) {
            int buffIdx = s * (2 * nchan) + 2 * ch;
            short real = buff[buffIdx];
            short imag = buff[buffIdx + 1];
            int complexIdx = ch * len + s;
            complexData[complexIdx] = make_float2((float)real, (float)imag);
        }
    }

    free(buff);
    *success = 1;
    return (float*)complexData;
}

float* read_data_cuda_dual_channel(FILE* fid, int nchan, int len, int* success, int ref_channel, int monitor_channel) {
    int totalInt16 = 2 * nchan * len;
    short* buff = (short*)malloc(totalInt16 * sizeof(short));
    if (!buff) {
        *success = 0;
        return NULL;
    }

    size_t itemsRead = fread(buff, sizeof(short), totalInt16, fid);
    if (itemsRead < totalInt16) {
        free(buff);
        *success = 0;
        return NULL;
    }

    int outputChannels = 2;
    float2* complexData = (float2*)malloc(outputChannels * len * sizeof(float2));
    if (!complexData) {
        free(buff);
        *success = 0;
        return NULL;
    }

    int channels[2] = {ref_channel, monitor_channel};
    for (int ch_idx = 0; ch_idx < 2; ch_idx++) {
        int ch = channels[ch_idx];
        for (int s = 0; s < len; s++) {
            int buffIdx = s * (2 * nchan) + 2 * ch;
            short real = buff[buffIdx];
            short imag = buff[buffIdx + 1];
            int complexIdx = ch_idx * len + s;
            complexData[complexIdx] = make_float2((float)real, (float)imag);
        }
    }

    free(buff);
    *success = 1;
    return (float*)complexData;
}

float* read_data_cuda_three_channel(FILE* fid, int nchan, int len, int* success, int ch0, int ch1, int ch2) {
    int totalInt16 = 2 * nchan * len;
    short* buff = (short*)malloc(totalInt16 * sizeof(short));
    if (!buff) {
        *success = 0;
        return NULL;
    }

    size_t itemsRead = fread(buff, sizeof(short), totalInt16, fid);
    if (itemsRead < totalInt16) {
        free(buff);
        *success = 0;
        return NULL;
    }

    int outputChannels = 3;
    float2* complexData = (float2*)malloc(outputChannels * len * sizeof(float2));
    if (!complexData) {
        free(buff);
        *success = 0;
        return NULL;
    }

    int channels[3] = {ch0, ch1, ch2};
    for (int ch_idx = 0; ch_idx < 3; ch_idx++) {
        int ch = channels[ch_idx];
        for (int s = 0; s < len; s++) {
            int buffIdx = s * (2 * nchan) + 2 * ch;
            short real = buff[buffIdx];
            short imag = buff[buffIdx + 1];
            int complexIdx = ch_idx * len + s;
            complexData[complexIdx] = make_float2((float)real, (float)imag);
        }
    }

    free(buff);
    *success = 1;
    return (float*)complexData;
}

float* batch_descend_sample_gpu(const float* h_src, int nchan, int srcLen, int newLen) {
    if (newLen <= 0 || nchan <= 0) return NULL;

    cufftComplex* h_dst = (cufftComplex*)malloc(nchan * newLen * sizeof(cufftComplex));
    if (!h_dst) return NULL;

    cufftComplex* d_src = NULL, * d_fft = NULL, * d_trunc = NULL, * d_ifft = NULL;

    size_t srcSize = nchan * srcLen * sizeof(cufftComplex);
    size_t dstSize = nchan * newLen * sizeof(cufftComplex);

    CUDA_CHECK(cudaMalloc(&d_src, srcSize));
    CUDA_CHECK(cudaMalloc(&d_fft, srcSize));
    CUDA_CHECK(cudaMalloc(&d_trunc, dstSize));
    CUDA_CHECK(cudaMalloc(&d_ifft, dstSize));

    CUDA_CHECK(cudaMemcpy(d_src, h_src, srcSize, cudaMemcpyHostToDevice));

    cufftHandle plan_forward, plan_inverse;
    int rank = 1;
    int fftSize = srcLen;
    int fftNewSize = newLen;
    int inembed[] = { fftSize };
    int onembed[] = { fftSize };
    int inembed_new[] = { fftNewSize };
    int onembed_new[] = { fftNewSize };
    int stride = 1;
    int dist = srcLen;
    int dist_new = newLen;

    CUFFT_CHECK(cufftPlanMany(&plan_forward, rank, &fftSize,
        inembed, stride, dist,
        onembed, stride, dist,
        CUFFT_C2C, nchan));

    CUFFT_CHECK(cufftPlanMany(&plan_inverse, rank, &fftNewSize,
        inembed_new, stride, dist_new,
        onembed_new, stride, dist_new,
        CUFFT_C2C, nchan));

    CUFFT_CHECK(cufftExecC2C(plan_forward, d_src, d_fft, CUFFT_FORWARD));

    int blockSize = 1024;
    for (int ch = 0; ch < nchan; ch++) {
        cufftComplex* ch_fft = d_fft + ch * srcLen;
        cufftComplex* ch_trunc = d_trunc + ch * newLen;

        int gridSize = (newLen + blockSize - 1) / blockSize;
        packFrequencyDomain <<<gridSize, blockSize>>> (ch_fft, ch_trunc, srcLen, newLen);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUFFT_CHECK(cufftExecC2C(plan_inverse, d_trunc, d_ifft, CUFFT_INVERSE));

    float normFactor = 1.0f / newLen;
    int totalSamples = nchan * newLen;
    int gridSize = (totalSamples + blockSize - 1) / blockSize;
    normalizeIFFT <<<gridSize, blockSize>>> (d_ifft, totalSamples, normFactor);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dst, d_ifft, dstSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_fft));
    CUDA_CHECK(cudaFree(d_trunc));
    CUDA_CHECK(cudaFree(d_ifft));
    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);

    return (float*)h_dst;
}

__global__ void packFrequencyDomain_batch(
    const cuFloatComplex* d_fftSrc, 
    cuFloatComplex* d_fftDst,
    int srcLen, int newLen, int nchan) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nchan * newLen;
    if (idx >= total) return;
    
    int ch = idx / newLen;
    int sample = idx % newLen;
    int halfNew = newLen / 2;
    
    if (sample <= halfNew) {
        d_fftDst[idx] = d_fftSrc[ch * srcLen + sample];
    } else {
        int srcIdx = ch * srcLen + srcLen - (newLen - sample);
        d_fftDst[idx] = d_fftSrc[srcIdx];
    }
}

FFTPlanCache* create_fft_plan_cache(int nchan, int srcLen, int dstLen) {
    if (nchan <= 0 || srcLen <= 0 || dstLen <= 0) {
        return NULL;
    }

    FFTPlanCache* cache = (FFTPlanCache*)malloc(sizeof(FFTPlanCache));
    if (!cache) {
        return NULL;
    }

    cache->srcLen = srcLen;
    cache->dstLen = dstLen;
    cache->nchan = nchan;
    cache->is_valid = false;
    cache->d_src = NULL;
    cache->d_fft = NULL;
    cache->d_trunc = NULL;
    cache->d_ifft = NULL;
    cache->d_raw_input = NULL;
    cache->d_complex_output = NULL;

    int rank = 1;
    int fftSize = srcLen;
    int fftNewSize = dstLen;
    int inembed[] = { fftSize };
    int onembed[] = { fftSize };
    int inembed_new[] = { fftNewSize };
    int onembed_new[] = { fftNewSize };
    int stride = 1;
    int dist = srcLen;
    int dist_new = dstLen;

    cufftResult err;
    err = cufftPlanMany(&cache->plan_forward, rank, &fftSize,
        inembed, stride, dist,
        onembed, stride, dist,
        CUFFT_C2C, nchan);
    if (err != CUFFT_SUCCESS) {
        free(cache);
        return NULL;
    }

    err = cufftPlanMany(&cache->plan_inverse, rank, &fftNewSize,
        inembed_new, stride, dist_new,
        onembed_new, stride, dist_new,
        CUFFT_C2C, nchan);
    if (err != CUFFT_SUCCESS) {
        cufftDestroy(cache->plan_forward);
        free(cache);
        return NULL;
    }

    size_t srcSize = nchan * srcLen * sizeof(cuFloatComplex);
    size_t dstSize = nchan * dstLen * sizeof(cuFloatComplex);
    size_t rawSize = 2 * nchan * srcLen * sizeof(short);
    
    CUDA_CHECK(cudaMalloc(&cache->d_src, srcSize));
    CUDA_CHECK(cudaMalloc(&cache->d_fft, srcSize));
    CUDA_CHECK(cudaMalloc(&cache->d_trunc, dstSize));
    CUDA_CHECK(cudaMalloc(&cache->d_ifft, dstSize));
    CUDA_CHECK(cudaMalloc(&cache->d_raw_input, rawSize));
    CUDA_CHECK(cudaMalloc(&cache->d_complex_output, dstSize));
    
    CUDA_CHECK(cudaStreamCreate(&cache->stream));

    cache->is_valid = true;
    return cache;
}

void destroy_fft_plan_cache(FFTPlanCache* cache) {
    if (!cache) {
        return;
    }
    if (cache->is_valid) {
        cufftDestroy(cache->plan_forward);
        cufftDestroy(cache->plan_inverse);
        
        if (cache->d_src) CUDA_CHECK(cudaFree(cache->d_src));
        if (cache->d_fft) CUDA_CHECK(cudaFree(cache->d_fft));
        if (cache->d_trunc) CUDA_CHECK(cudaFree(cache->d_trunc));
        if (cache->d_ifft) CUDA_CHECK(cudaFree(cache->d_ifft));
        if (cache->d_raw_input) CUDA_CHECK(cudaFree(cache->d_raw_input));
        if (cache->d_complex_output) CUDA_CHECK(cudaFree(cache->d_complex_output));
        
        CUDA_CHECK(cudaStreamDestroy(cache->stream));
    }
    free(cache);
}

float* batch_descend_sample_gpu_with_cache(const float* h_src, int nchan, int srcLen, int newLen, FFTPlanCache* cache) {
    if (newLen <= 0 || nchan <= 0 || !cache || !cache->is_valid) {
        return NULL;
    }

    if (cache->nchan != nchan || cache->srcLen != srcLen || cache->dstLen != newLen) {
        return NULL;
    }

    cufftComplex* h_dst = (cufftComplex*)malloc(nchan * newLen * sizeof(cufftComplex));
    if (!h_dst) return NULL;

    size_t srcSize = nchan * srcLen * sizeof(cufftComplex);
    size_t dstSize = nchan * newLen * sizeof(cufftComplex);

    CUDA_CHECK(cudaMemcpyAsync(cache->d_src, h_src, srcSize, cudaMemcpyHostToDevice, cache->stream));

    CUFFT_CHECK(cufftExecC2C(cache->plan_forward, cache->d_src, cache->d_fft, CUFFT_FORWARD));

    int blockSize = 256;
    int totalSamples = nchan * newLen;
    int gridSize = (totalSamples + blockSize - 1) / blockSize;
    packFrequencyDomain_batch<<<gridSize, blockSize, 0, cache->stream>>>(cache->d_fft, cache->d_trunc, srcLen, newLen, nchan);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(cache->plan_inverse, cache->d_trunc, cache->d_ifft, CUFFT_INVERSE));

    float normFactor = 1.0f / newLen;
    gridSize = (totalSamples + blockSize - 1) / blockSize;
    normalizeIFFT<<<gridSize, blockSize, 0, cache->stream>>>(cache->d_ifft, totalSamples, normFactor);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(h_dst, cache->d_ifft, dstSize, cudaMemcpyDeviceToHost, cache->stream));
    CUDA_CHECK(cudaStreamSynchronize(cache->stream));

    return (float*)h_dst;
}

short* cuda_downsample_data_with_cache(FILE* fid, int nchan, int len,
    int targetFs, int originalFs,
    size_t* outputSize, int downsampleAllChannels, FFTPlanCache* cache) {
    if (!fid || nchan <= 0 || len <= 0 || targetFs <= 0 || originalFs <= 0 || !cache || !cache->is_valid) {
        return NULL;
    }

    int readSuccess;
    float* rawData = read_data_cuda(fid, nchan, len, &readSuccess, downsampleAllChannels);
    if (!readSuccess || !rawData) {
        return NULL;
    }

    int newLen = (int)floor(len * (double)targetFs / originalFs);
    if (newLen <= 0) {
        free(rawData);
        return NULL;
    }

    float* processedData = NULL;
    int outputChannels = downsampleAllChannels ? nchan : 1;
    
    if (downsampleAllChannels) {
        processedData = batch_descend_sample_gpu_with_cache(rawData, nchan, len, newLen, cache);
    } else {
        processedData = batch_descend_sample_gpu_with_cache(rawData, 1, len, newLen, cache);
    }
    free(rawData);

    if (!processedData) {
        return NULL;
    }

    *outputSize = 2 * outputChannels * newLen * sizeof(short);
    short* output = (short*)malloc(*outputSize);
    if (!output) {
        free(processedData);
        return NULL;
    }

    float2* procData = (float2*)processedData;
    for (int s = 0; s < newLen; s++) {
        for (int ch = 0; ch < outputChannels; ch++) {
            float2 val = procData[ch * newLen + s];
            int outIdx = s * (2 * outputChannels) + 2 * ch;
            output[outIdx] = (short)val.x;
            output[outIdx + 1] = (short)val.y;
        }
    }

    free(processedData);
    return output;
}

short* cuda_downsample_data(FILE* fid, int nchan, int len,
    int targetFs, int originalFs,
    size_t* outputSize, int downsampleAllChannels) {
    if (!fid || nchan <= 0 || len <= 0 || targetFs <= 0 || originalFs <= 0) {
        return NULL;
    }

    int readSuccess;
    float* rawData = read_data_cuda(fid, nchan, len, &readSuccess, downsampleAllChannels);
    if (!readSuccess || !rawData) {
        return NULL;
    }

    int newLen = (int)floor(len * (double)targetFs / originalFs);
    if (newLen <= 0) {
        free(rawData);
        return NULL;
    }

    float* processedData = NULL;
    int outputChannels = downsampleAllChannels ? nchan : 1;
    
    if (downsampleAllChannels) {
        processedData = batch_descend_sample_gpu(rawData, nchan, len, newLen);
    } else {
        processedData = batch_descend_sample_gpu(rawData, 1, len, newLen);
    }
    free(rawData);

    if (!processedData) {
        return NULL;
    }

    *outputSize = 2 * outputChannels * newLen * sizeof(short);
    short* output = (short*)malloc(*outputSize);
    if (!output) {
        free(processedData);
        return NULL;
    }

    float2* procData = (float2*)processedData;
    for (int s = 0; s < newLen; s++) {
        for (int ch = 0; ch < outputChannels; ch++) {
            float2 val = procData[ch * newLen + s];
            int outIdx = s * (2 * outputChannels) + 2 * ch;
            output[outIdx] = (short)val.x;
            output[outIdx + 1] = (short)val.y;
        }
    }

    free(processedData);
    return output;
}

// 新增：直接输出cufftComplex格式
cufftComplex* cuda_downsample_data_complex_with_cache(FILE* fid, int nchan, int len,
    int targetFs, int originalFs,
    int* outputSamples, int downsampleAllChannels, FFTPlanCache* cache) {
    if (!fid || nchan <= 0 || len <= 0 || targetFs <= 0 || originalFs <= 0 || !cache || !cache->is_valid) {
        return NULL;
    }

    int readSuccess;
    float* rawData = read_data_cuda(fid, nchan, len, &readSuccess, downsampleAllChannels);
    if (!readSuccess || !rawData) {
        return NULL;
    }

    int newLen = (int)floor(len * (double)targetFs / originalFs);
    if (newLen <= 0) {
        free(rawData);
        return NULL;
    }

    float* processedData = NULL;
    int outputChannels = downsampleAllChannels ? nchan : 1;
    
    if (downsampleAllChannels) {
        processedData = batch_descend_sample_gpu_with_cache(rawData, nchan, len, newLen, cache);
    } else {
        processedData = batch_descend_sample_gpu_with_cache(rawData, 1, len, newLen, cache);
    }
    free(rawData);

    if (!processedData) {
        return NULL;
    }

    *outputSamples = outputChannels * newLen;
    cufftComplex* output = (cufftComplex*)malloc(*outputSamples * sizeof(cufftComplex));
    if (!output) {
        free(processedData);
        return NULL;
    }

    int total_elements = outputChannels * newLen;
    float2* procData = (float2*)processedData;
    
    for (int idx = 0; idx < total_elements; idx++) {
        float2 val = procData[idx];
        output[idx].x = val.x;
        output[idx].y = val.y;
    }

    free(processedData);
    return output;
}

cufftComplex* cuda_downsample_data_complex_dual_channel_with_cache(FILE* fid, int nchan, int len,
    int targetFs, int originalFs,
    int* outputSamples, int ref_channel, int monitor_channel, FFTPlanCache* cache) {
    if (!fid || nchan <= 0 || len <= 0 || targetFs <= 0 || originalFs <= 0 || !cache || !cache->is_valid) {
        return NULL;
    }

    int readSuccess;
    float* rawData = read_data_cuda_dual_channel(fid, nchan, len, &readSuccess, ref_channel, monitor_channel);
    if (!readSuccess || !rawData) {
        return NULL;
    }

    int newLen = (int)floor(len * (double)targetFs / originalFs);
    if (newLen <= 0) {
        free(rawData);
        return NULL;
    }

    float* processedData = batch_descend_sample_gpu_with_cache(rawData, 2, len, newLen, cache);
    free(rawData);

    if (!processedData) {
        return NULL;
    }

    *outputSamples = 2 * newLen;
    cufftComplex* output = (cufftComplex*)malloc(*outputSamples * sizeof(cufftComplex));
    if (!output) {
        free(processedData);
        return NULL;
    }

    int total_elements = 2 * newLen;
    float2* procData = (float2*)processedData;
    
    for (int idx = 0; idx < total_elements; idx++) {
        float2 val = procData[idx];
        output[idx].x = val.x;
        output[idx].y = val.y;
    }

    free(processedData);
    return output;
}

__global__ void convert_float2_to_cufftComplex_kernel(
    const float* __restrict__ d_float2,
    cuFloatComplex* d_cufft,
    int n, int nchan) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_samples = nchan * n;
    if (idx < total_samples) {
        float real = d_float2[2 * idx + 0];
        float imag = d_float2[2 * idx + 1];
        d_cufft[idx] = make_cuFloatComplex(real, imag);
    }
}

__global__ void copy_cuFloatComplex_to_cufftComplex_kernel(
    const cuFloatComplex* __restrict__ d_src,
    cufftComplex* d_dst,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_dst[idx].x = cuCrealf(d_src[idx]);
        d_dst[idx].y = cuCimagf(d_src[idx]);
    }
}

cufftComplex* cuda_downsample_data_complex_dual_channel_device_only_with_cache(
    FILE* fid, int nchan, int len,
    int targetFs, int originalFs,
    int* outputSamples, int ref_channel, int monitor_channel,
    FFTPlanCache* cache) {
    if (!fid || nchan <= 0 || len <= 0 || targetFs <= 0 || originalFs <= 0 || !cache || !cache->is_valid) {
        return NULL;
    }

    int readSuccess;
    float* rawData = read_data_cuda_dual_channel(fid, nchan, len, &readSuccess, ref_channel, monitor_channel);
    if (!readSuccess || !rawData) {
        return NULL;
    }

    int newLen = (int)floor(len * (double)targetFs / originalFs);
    if (newLen <= 0) {
        free(rawData);
        return NULL;
    }

    size_t float2Size = 2 * 2 * len * sizeof(float);
    float* d_float2 = NULL;
    CUDA_CHECK(cudaMalloc(&d_float2, float2Size));
    CUDA_CHECK(cudaMemcpyAsync(d_float2, rawData, float2Size, cudaMemcpyHostToDevice, cache->stream));
    free(rawData);

    int blockSize = 256;
    int convertGridSize = (2 * len + blockSize - 1) / blockSize;
    convert_float2_to_cufftComplex_kernel<<<convertGridSize, blockSize, 0, cache->stream>>>(
        d_float2, cache->d_src, len, 2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_float2));

    CUFFT_CHECK(cufftExecC2C(cache->plan_forward, cache->d_src, cache->d_fft, CUFFT_FORWARD));

    int totalSamples = 2 * newLen;
    int gridSize = (totalSamples + blockSize - 1) / blockSize;
    packFrequencyDomain_batch<<<gridSize, blockSize, 0, cache->stream>>>(cache->d_fft, cache->d_trunc, len, newLen, 2);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(cache->plan_inverse, cache->d_trunc, cache->d_ifft, CUFFT_INVERSE));

    float normFactor = 1.0f / newLen;
    gridSize = (totalSamples + blockSize - 1) / blockSize;
    normalizeIFFT<<<gridSize, blockSize, 0, cache->stream>>>(cache->d_ifft, totalSamples, normFactor);
    CUDA_CHECK(cudaGetLastError());

    copy_cuFloatComplex_to_cufftComplex_kernel<<<gridSize, blockSize, 0, cache->stream>>>(
        cache->d_ifft, cache->d_complex_output, totalSamples);
    CUDA_CHECK(cudaGetLastError());

    *outputSamples = 2 * newLen;
    return cache->d_complex_output;
}

cufftComplex* cuda_downsample_data_complex_three_channel_device_only_with_cache(
    FILE* fid, int nchan, int len,
    int targetFs, int originalFs,
    int* outputSamples, int ch0, int ch1, int ch2,
    FFTPlanCache* cache) {
    if (!fid || nchan <= 0 || len <= 0 || targetFs <= 0 || originalFs <= 0 || !cache || !cache->is_valid) {
        return NULL;
    }

    int readSuccess;
    float* rawData = read_data_cuda_three_channel(fid, nchan, len, &readSuccess, ch0, ch1, ch2);
    if (!readSuccess || !rawData) {
        return NULL;
    }

    int newLen = (int)floor(len * (double)targetFs / originalFs);
    if (newLen <= 0) {
        free(rawData);
        return NULL;
    }

    size_t float2Size = 2 * 3 * len * sizeof(float);
    float* d_float2 = NULL;
    CUDA_CHECK(cudaMalloc(&d_float2, float2Size));
    CUDA_CHECK(cudaMemcpyAsync(d_float2, rawData, float2Size, cudaMemcpyHostToDevice, cache->stream));
    free(rawData);

    int blockSize = 256;
    int convertGridSize = (3 * len + blockSize - 1) / blockSize;
    convert_float2_to_cufftComplex_kernel<<<convertGridSize, blockSize, 0, cache->stream>>>(
        d_float2, cache->d_src, len, 3);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_float2));

    CUFFT_CHECK(cufftExecC2C(cache->plan_forward, cache->d_src, cache->d_fft, CUFFT_FORWARD));

    int totalSamples = 3 * newLen;
    int gridSize = (totalSamples + blockSize - 1) / blockSize;
    packFrequencyDomain_batch<<<gridSize, blockSize, 0, cache->stream>>>(cache->d_fft, cache->d_trunc, len, newLen, 3);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(cache->plan_inverse, cache->d_trunc, cache->d_ifft, CUFFT_INVERSE));

    float normFactor = 1.0f / newLen;
    gridSize = (totalSamples + blockSize - 1) / blockSize;
    normalizeIFFT<<<gridSize, blockSize, 0, cache->stream>>>(cache->d_ifft, totalSamples, normFactor);
    CUDA_CHECK(cudaGetLastError());

    copy_cuFloatComplex_to_cufftComplex_kernel<<<gridSize, blockSize, 0, cache->stream>>>(
        cache->d_ifft, cache->d_complex_output, totalSamples);
    CUDA_CHECK(cudaGetLastError());

    *outputSamples = 3 * newLen;
    return cache->d_complex_output;
}
