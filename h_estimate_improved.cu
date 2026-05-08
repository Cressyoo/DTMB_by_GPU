#include "h_estimate_improved.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>

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
            fprintf(stderr, "CUFFT error in %s at line %d: error code %d\n", \
                    __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void set_last_indices_zero_batch_kernel(cuFloatComplex* d_hr_batch, int frame_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = frame_num * 255;
    if (idx < total_elements) {
        int fi = idx / 255;
        int i = idx % 255;
        if (i >= 250 && i < 255) {
            d_hr_batch[idx] = make_cuFloatComplex(0.0f, 0.0f);
        }
    }
}

__global__ void complex_conj_multiply_batch_kernel(const cuFloatComplex* d_a, const cuFloatComplex* d_b,
                                                      cuFloatComplex* d_out, int frame_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = frame_num * 255;
    if (idx < total_elements) {
        cuFloatComplex a = d_a[idx];
        cuFloatComplex b = d_b[idx];
        float re_a = cuCrealf(a);
        float im_a = cuCimagf(a);
        float re_b = cuCrealf(b);
        float im_b = cuCimagf(b);
        float real_part = re_a * re_b + im_a * im_b;
        float imag_part = im_a * re_b - re_a * im_b;
        d_out[idx] = make_cuFloatComplex(real_part, imag_part);
    }
}

HEstimateImprovedGPUCache* create_h_estimate_improved_gpu_cache_with_batch(int max_frame_num) {
    HEstimateImprovedGPUCache* cache = (HEstimateImprovedGPUCache*)malloc(sizeof(HEstimateImprovedGPUCache));
    if (!cache) {
        return NULL;
    }

    cache->max_len = 255;
    cache->max_frame_num = max_frame_num;
    cache->is_valid = false;

    size_t complexBatchSize = max_frame_num * 255 * sizeof(cuFloatComplex);

    CUDA_CHECK(cudaMalloc(&cache->d_dat255_batch, complexBatchSize));
    CUDA_CHECK(cudaMalloc(&cache->d_pn255_batch, complexBatchSize));
    CUDA_CHECK(cudaMalloc(&cache->d_hr_batch, complexBatchSize));
    CUDA_CHECK(cudaMalloc(&cache->d_fft_dat_batch, complexBatchSize));
    CUDA_CHECK(cudaMalloc(&cache->d_fft_pn_batch, complexBatchSize));
    CUDA_CHECK(cudaMalloc(&cache->d_conj_mul_result_batch, complexBatchSize));

    int rank = 1;
    int fft_size = 255;
    int inembed[] = { fft_size };
    int onembed[] = { fft_size };
    int stride = 1;
    int dist = fft_size;
    int batch_size = max_frame_num;

    CUFFT_CHECK(cufftPlanMany(&cache->fft_plan_batch, rank, &fft_size,
        inembed, stride, dist,
        onembed, stride, dist,
        CUFFT_C2C, batch_size));
    CUFFT_CHECK(cufftPlanMany(&cache->ifft_plan_batch, rank, &fft_size,
        inembed, stride, dist,
        onembed, stride, dist,
        CUFFT_C2C, batch_size));

    cache->is_valid = true;
    return cache;
}

void destroy_h_estimate_improved_gpu_cache(HEstimateImprovedGPUCache* cache) {
    if (!cache) {
        return;
    }
    if (cache->is_valid) {
        CUDA_CHECK(cudaFree(cache->d_dat255_batch));
        CUDA_CHECK(cudaFree(cache->d_pn255_batch));
        CUDA_CHECK(cudaFree(cache->d_hr_batch));
        CUDA_CHECK(cudaFree(cache->d_fft_dat_batch));
        CUDA_CHECK(cudaFree(cache->d_fft_pn_batch));
        CUDA_CHECK(cudaFree(cache->d_conj_mul_result_batch));
        CUFFT_CHECK(cufftDestroy(cache->fft_plan_batch));
        CUFFT_CHECK(cufftDestroy(cache->ifft_plan_batch));
    }
    free(cache);
}

void h_estimate_improved_gpu_batch_with_cache(const std::vector<cuFloatComplex>& dat255_batch,
                                                const std::vector<cuFloatComplex>& pn255_batch,
                                                int frame_num,
                                                std::vector<cuFloatComplex>& hr_batch,
                                                HEstimateImprovedGPUCache* cache) {
    if (!cache || !cache->is_valid || cache->max_frame_num < frame_num) {
        return;
    }

    size_t complexBatchSize = frame_num * 255 * sizeof(cuFloatComplex);

    CUDA_CHECK(cudaMemcpy(cache->d_dat255_batch, dat255_batch.data(), complexBatchSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cache->d_pn255_batch, pn255_batch.data(), complexBatchSize, cudaMemcpyHostToDevice));

    CUFFT_CHECK(cufftExecC2C(cache->fft_plan_batch, (cufftComplex*)cache->d_dat255_batch, (cufftComplex*)cache->d_fft_dat_batch, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(cache->fft_plan_batch, (cufftComplex*)cache->d_pn255_batch, (cufftComplex*)cache->d_fft_pn_batch, CUFFT_FORWARD));

    int blockSize = 256;
    int gridSize = (frame_num * 255 + blockSize - 1) / blockSize;
    complex_conj_multiply_batch_kernel<<<gridSize, blockSize>>>(cache->d_fft_dat_batch, cache->d_fft_pn_batch, cache->d_conj_mul_result_batch, frame_num);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(cache->ifft_plan_batch, (cufftComplex*)cache->d_conj_mul_result_batch, (cufftComplex*)cache->d_hr_batch, CUFFT_INVERSE));

    set_last_indices_zero_batch_kernel<<<gridSize, blockSize>>>(cache->d_hr_batch, frame_num);
    CUDA_CHECK(cudaGetLastError());

    hr_batch.resize(frame_num * 255);
    CUDA_CHECK(cudaMemcpy(hr_batch.data(), cache->d_hr_batch, complexBatchSize, cudaMemcpyDeviceToHost));

    const int M = 255;
    float scale = 1.0f / (float)M / (float)M;
    for (int fi = 0; fi < frame_num; fi++) {
        std::vector<std::complex<float>> hr_raw_frame(255);
        float max_val = 0.0f;
        for (int i = 0; i < 255; i++) {
            int idx = fi * 255 + i;
            cuFloatComplex val = hr_batch[idx];
            float real = cuCrealf(val) * scale;
            float imag = cuCimagf(val) * scale;
            hr_raw_frame[i] = std::complex<float>(real, imag);
            float abs_val = sqrtf(real * real + imag * imag);
            if (abs_val > max_val) {
                max_val = abs_val;
            }
        }

        float threshold = 0.05f * max_val;
        int n = 0;
        std::complex<float> sum_RPr(0.0f, 0.0f);
        for (int i = 0; i < 255; i++) {
            if (std::abs(hr_raw_frame[i]) >= threshold) {
                n++;
                sum_RPr += hr_raw_frame[i];
            }
        }

        std::complex<float> correction(0.0f, 0.0f);
        int denom = M + 1 - n;
        if (denom > 0) {
            correction = sum_RPr / (float)denom;
        }

        for (int i = 0; i < 255; i++) {
            int idx = fi * 255 + i;
            std::complex<float> hr_val = hr_raw_frame[i] + correction;
            hr_batch[idx] = make_cuFloatComplex(hr_val.real(), hr_val.imag());
        }
    }
}

void h_estimate_improved_gpu_batch_device_with_cache(
                                               int frame_num,
                                               std::vector<cuFloatComplex>& hr_batch,
                                               HEstimateImprovedGPUCache* cache) {
    if (!cache || !cache->is_valid || cache->max_frame_num < frame_num) {
        return;
    }

    CUFFT_CHECK(cufftExecC2C(cache->fft_plan_batch, (cufftComplex*)cache->d_dat255_batch, (cufftComplex*)cache->d_fft_dat_batch, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(cache->fft_plan_batch, (cufftComplex*)cache->d_pn255_batch, (cufftComplex*)cache->d_fft_pn_batch, CUFFT_FORWARD));

    int blockSize = 256;
    int gridSize = (frame_num * 255 + blockSize - 1) / blockSize;
    complex_conj_multiply_batch_kernel<<<gridSize, blockSize>>>(cache->d_fft_dat_batch, cache->d_fft_pn_batch, cache->d_conj_mul_result_batch, frame_num);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(cache->ifft_plan_batch, (cufftComplex*)cache->d_conj_mul_result_batch, (cufftComplex*)cache->d_hr_batch, CUFFT_INVERSE));

    set_last_indices_zero_batch_kernel<<<gridSize, blockSize>>>(cache->d_hr_batch, frame_num);
    CUDA_CHECK(cudaGetLastError());

    size_t complexBatchSize = frame_num * 255 * sizeof(cuFloatComplex);
    hr_batch.resize(frame_num * 255);
    CUDA_CHECK(cudaMemcpy(hr_batch.data(), cache->d_hr_batch, complexBatchSize, cudaMemcpyDeviceToHost));

    const int M = 255;
    float scale = 1.0f / (float)M / (float)M;
    for (int fi = 0; fi < frame_num; fi++) {
        std::vector<std::complex<float>> hr_raw_frame(255);
        float max_val = 0.0f;
        for (int i = 0; i < 255; i++) {
            int idx = fi * 255 + i;
            cuFloatComplex val = hr_batch[idx];
            float real = cuCrealf(val) * scale;
            float imag = cuCimagf(val) * scale;
            hr_raw_frame[i] = std::complex<float>(real, imag);
            float abs_val = sqrtf(real * real + imag * imag);
            if (abs_val > max_val) {
                max_val = abs_val;
            }
        }

        float threshold = 0.05f * max_val;
        int n = 0;
        std::complex<float> sum_RPr(0.0f, 0.0f);
        for (int i = 0; i < 255; i++) {
            if (std::abs(hr_raw_frame[i]) >= threshold) {
                n++;
                sum_RPr += hr_raw_frame[i];
            }
        }

        std::complex<float> correction(0.0f, 0.0f);
        int denom = M + 1 - n;
        if (denom > 0) {
            correction = sum_RPr / (float)denom;
        }

        for (int i = 0; i < 255; i++) {
            int idx = fi * 255 + i;
            std::complex<float> hr_val = hr_raw_frame[i] + correction;
            hr_batch[idx] = make_cuFloatComplex(hr_val.real(), hr_val.imag());
        }
    }
}
