// h_estimate_f.cu - Frequency domain channel estimation implementation
#include "h_estimate_f.h"
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

__global__ void complex_divide_kernel(const cuFloatComplex* d_a, const cuFloatComplex* d_b,
                                        cuFloatComplex* d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        cuFloatComplex a = d_a[idx];
        cuFloatComplex b = d_b[idx];
        
        float real_b = cuCrealf(b);
        float imag_b = cuCimagf(b);
        float denom = real_b * real_b + imag_b * imag_b;
        
        if (denom > 1e-10f) {
            float real_a = cuCrealf(a);
            float imag_a = cuCimagf(a);
            
            float out_real = (real_a * real_b + imag_a * imag_b) / denom;
            float out_imag = (imag_a * real_b - real_a * imag_b) / denom;
            
            d_out[idx] = make_cuFloatComplex(out_real, out_imag);
        } else {
            d_out[idx] = make_cuFloatComplex(0.0f, 0.0f);
        }
    }
}

__global__ void set_last_indices_zero_kernel(cuFloatComplex* d_hf) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 250 && idx < 255) {
        d_hf[idx] = make_cuFloatComplex(0.0f, 0.0f);
    }
}

__global__ void set_last_indices_zero_batch_kernel(cuFloatComplex* d_hf_batch, int frame_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = frame_num * 255;
    if (idx < total_elements) {
        int fi = idx / 255;
        int i = idx % 255;
        if (i >= 250 && i < 255) {
            d_hf_batch[idx] = make_cuFloatComplex(0.0f, 0.0f);
        }
    }
}

__global__ void complex_divide_batch_kernel(const cuFloatComplex* d_a, const cuFloatComplex* d_b,
                                               cuFloatComplex* d_out, int frame_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = frame_num * 255;
    if (idx < total_elements) {
        int fi = idx / 255;
        int i = idx % 255;
        
        cuFloatComplex a = d_a[fi * 255 + i];
        cuFloatComplex b = d_b[fi * 255 + i];
        
        float real_b = cuCrealf(b);
        float imag_b = cuCimagf(b);
        float denom = real_b * real_b + imag_b * imag_b;
        
        if (denom > 1e-10f) {
            float real_a = cuCrealf(a);
            float imag_a = cuCimagf(a);
            
            float out_real = (real_a * real_b + imag_a * imag_b) / denom;
            float out_imag = (imag_a * real_b - real_a * imag_b) / denom;
            
            d_out[idx] = make_cuFloatComplex(out_real, out_imag);
        } else {
            d_out[idx] = make_cuFloatComplex(0.0f, 0.0f);
        }
    }
}

HEstimateFGPUCache* create_h_estimate_f_gpu_cache() {
    HEstimateFGPUCache* cache = (HEstimateFGPUCache*)malloc(sizeof(HEstimateFGPUCache));
    if (!cache) {
        return NULL;
    }

    cache->max_len = 255;
    cache->max_frame_num = 0;
    cache->is_valid = false;

    size_t complexSize = 255 * sizeof(cuFloatComplex);
    size_t floatSize = 255 * sizeof(float);

    CUDA_CHECK(cudaMalloc(&cache->d_dat255, complexSize));
    CUDA_CHECK(cudaMalloc(&cache->d_pn255, complexSize));
    CUDA_CHECK(cudaMalloc(&cache->d_hf, complexSize));
    CUDA_CHECK(cudaMalloc(&cache->d_abs_hf, floatSize));

    CUFFT_CHECK(cufftPlan1d(&cache->fft_plan, 255, CUFFT_C2C, 1));
    CUFFT_CHECK(cufftPlan1d(&cache->ifft_plan, 255, CUFFT_C2C, 1));

    cache->d_dat255_batch = NULL;
    cache->d_pn255_batch = NULL;
    cache->d_hf_batch = NULL;
    cache->d_fft_dat_batch = NULL;
    cache->d_fft_pn_batch = NULL;
    cache->d_div_result_batch = NULL;

    cache->is_valid = true;
    return cache;
}

HEstimateFGPUCache* create_h_estimate_f_gpu_cache_with_batch(int max_frame_num) {
    HEstimateFGPUCache* cache = (HEstimateFGPUCache*)malloc(sizeof(HEstimateFGPUCache));
    if (!cache) {
        return NULL;
    }

    cache->max_len = 255;
    cache->max_frame_num = max_frame_num;
    cache->is_valid = false;

    size_t complexSize = 255 * sizeof(cuFloatComplex);
    size_t floatSize = 255 * sizeof(float);
    size_t complexBatchSize = max_frame_num * 255 * sizeof(cuFloatComplex);

    CUDA_CHECK(cudaMalloc(&cache->d_dat255, complexSize));
    CUDA_CHECK(cudaMalloc(&cache->d_pn255, complexSize));
    CUDA_CHECK(cudaMalloc(&cache->d_hf, complexSize));
    CUDA_CHECK(cudaMalloc(&cache->d_abs_hf, floatSize));

    CUFFT_CHECK(cufftPlan1d(&cache->fft_plan, 255, CUFFT_C2C, 1));
    CUFFT_CHECK(cufftPlan1d(&cache->ifft_plan, 255, CUFFT_C2C, 1));

    CUDA_CHECK(cudaMalloc(&cache->d_dat255_batch, complexBatchSize));
    CUDA_CHECK(cudaMalloc(&cache->d_pn255_batch, complexBatchSize));
    CUDA_CHECK(cudaMalloc(&cache->d_hf_batch, complexBatchSize));
    CUDA_CHECK(cudaMalloc(&cache->d_fft_dat_batch, complexBatchSize));
    CUDA_CHECK(cudaMalloc(&cache->d_fft_pn_batch, complexBatchSize));
    CUDA_CHECK(cudaMalloc(&cache->d_div_result_batch, complexBatchSize));

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

void destroy_h_estimate_f_gpu_cache(HEstimateFGPUCache* cache) {
    if (!cache) {
        return;
    }
    if (cache->is_valid) {
        CUDA_CHECK(cudaFree(cache->d_dat255));
        CUDA_CHECK(cudaFree(cache->d_pn255));
        CUDA_CHECK(cudaFree(cache->d_hf));
        CUDA_CHECK(cudaFree(cache->d_abs_hf));
        CUFFT_CHECK(cufftDestroy(cache->fft_plan));
        CUFFT_CHECK(cufftDestroy(cache->ifft_plan));

        if (cache->max_frame_num > 0) {
            CUDA_CHECK(cudaFree(cache->d_dat255_batch));
            CUDA_CHECK(cudaFree(cache->d_pn255_batch));
            CUDA_CHECK(cudaFree(cache->d_hf_batch));
            CUDA_CHECK(cudaFree(cache->d_fft_dat_batch));
            CUDA_CHECK(cudaFree(cache->d_fft_pn_batch));
            CUDA_CHECK(cudaFree(cache->d_div_result_batch));
            CUFFT_CHECK(cufftDestroy(cache->fft_plan_batch));
            CUFFT_CHECK(cufftDestroy(cache->ifft_plan_batch));
        }
    }
    free(cache);
}

void h_estimate_f_gpu_batch_with_cache(const std::vector<cuFloatComplex>& dat255_batch,
                                        const std::vector<cuFloatComplex>& pn255_batch,
                                        int frame_num,
                                        std::vector<cuFloatComplex>& hf_batch,
                                        HEstimateFGPUCache* cache) {
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
    complex_divide_batch_kernel<<<gridSize, blockSize>>>(cache->d_fft_dat_batch, cache->d_fft_pn_batch, cache->d_div_result_batch, frame_num);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(cache->ifft_plan_batch, (cufftComplex*)cache->d_div_result_batch, (cufftComplex*)cache->d_hf_batch, CUFFT_INVERSE));

    set_last_indices_zero_batch_kernel<<<gridSize, blockSize>>>(cache->d_hf_batch, frame_num);
    CUDA_CHECK(cudaGetLastError());

    hf_batch.resize(frame_num * 255);
    CUDA_CHECK(cudaMemcpy(hf_batch.data(), cache->d_hf_batch, complexBatchSize, cudaMemcpyDeviceToHost));

    float scale = 1.0f / 255.0f;
    for (int fi = 0; fi < frame_num; fi++) {
        float max_val = 0.0f;
        for (int i = 0; i < 255; i++) {
            int idx = fi * 255 + i;
            cuFloatComplex val = hf_batch[idx];
            float real = cuCrealf(val) * scale;
            float imag = cuCimagf(val) * scale;
            float abs_val = sqrtf(real * real + imag * imag);
            if (abs_val > max_val) {
                max_val = abs_val;
            }
        }
        float threshold = 0.03f * max_val;
        for (int i = 0; i < 255; i++) {
            int idx = fi * 255 + i;
            cuFloatComplex val = hf_batch[idx];
            float real = cuCrealf(val) * scale;
            float imag = cuCimagf(val) * scale;
            float abs_val = sqrtf(real * real + imag * imag);
            if (abs_val < threshold) {
                hf_batch[idx] = make_cuFloatComplex(0.0f, 0.0f);
            } else {
                hf_batch[idx] = make_cuFloatComplex(real, imag);
            }
        }
    }
}

HEstimateFResult h_estimate_f_gpu(const std::vector<std::complex<float>>& dat255,
                                   const std::vector<std::complex<float>>& pn255) {
    HEstimateFResult result;
    result.hf.resize(255);
    result.hf_raw.resize(255);

    cuFloatComplex* d_dat255 = NULL;
    cuFloatComplex* d_pn255 = NULL;
    cuFloatComplex* d_fft_dat = NULL;
    cuFloatComplex* d_fft_pn = NULL;
    cuFloatComplex* d_div_result = NULL;
    cuFloatComplex* d_hf = NULL;

    size_t complexSize = 255 * sizeof(cuFloatComplex);

    CUDA_CHECK(cudaMalloc(&d_dat255, complexSize));
    CUDA_CHECK(cudaMalloc(&d_pn255, complexSize));
    CUDA_CHECK(cudaMalloc(&d_fft_dat, complexSize));
    CUDA_CHECK(cudaMalloc(&d_fft_pn, complexSize));
    CUDA_CHECK(cudaMalloc(&d_div_result, complexSize));
    CUDA_CHECK(cudaMalloc(&d_hf, complexSize));

    std::vector<cuFloatComplex> h_dat255(255);
    std::vector<cuFloatComplex> h_pn255(255);
    for (int i = 0; i < 255; i++) {
        h_dat255[i] = make_cuFloatComplex(dat255[i].real(), dat255[i].imag());
        h_pn255[i] = make_cuFloatComplex(pn255[i].real(), pn255[i].imag());
    }

    CUDA_CHECK(cudaMemcpy(d_dat255, h_dat255.data(), complexSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pn255, h_pn255.data(), complexSize, cudaMemcpyHostToDevice));

    cufftHandle fft_plan, ifft_plan;
    CUFFT_CHECK(cufftPlan1d(&fft_plan, 255, CUFFT_C2C, 1));
    CUFFT_CHECK(cufftPlan1d(&ifft_plan, 255, CUFFT_C2C, 1));

    CUFFT_CHECK(cufftExecC2C(fft_plan, (cufftComplex*)d_dat255, (cufftComplex*)d_fft_dat, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(fft_plan, (cufftComplex*)d_pn255, (cufftComplex*)d_fft_pn, CUFFT_FORWARD));

    int blockSize = 256;
    int gridSize = (255 + blockSize - 1) / blockSize;
    complex_divide_kernel<<<gridSize, blockSize>>>(d_fft_dat, d_fft_pn, d_div_result, 255);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(ifft_plan, (cufftComplex*)d_div_result, (cufftComplex*)d_hf, CUFFT_INVERSE));

    set_last_indices_zero_kernel<<<gridSize, blockSize>>>(d_hf);
    CUDA_CHECK(cudaGetLastError());

    std::vector<cuFloatComplex> h_hf(255);
    CUDA_CHECK(cudaMemcpy(h_hf.data(), d_hf, complexSize, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 255; i++) {
        float scale = 1.0f / 255.0f;
        result.hf_raw[i] = std::complex<float>(cuCrealf(h_hf[i]) * scale, cuCimagf(h_hf[i]) * scale);
    }

    result.hf = result.hf_raw;

    float max_val = 0.0f;
    for (int i = 0; i < 255; i++) {
        float abs_val = std::abs(result.hf[i]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }

    float threshold = 0.03f * max_val;
    for (int i = 0; i < 255; i++) {
        float abs_val = std::abs(result.hf[i]);
        if (abs_val < threshold) {
            result.hf[i] = std::complex<float>(0.0f, 0.0f);
        }
    }

    CUDA_CHECK(cudaFree(d_dat255));
    CUDA_CHECK(cudaFree(d_pn255));
    CUDA_CHECK(cudaFree(d_fft_dat));
    CUDA_CHECK(cudaFree(d_fft_pn));
    CUDA_CHECK(cudaFree(d_div_result));
    CUDA_CHECK(cudaFree(d_hf));
    CUFFT_CHECK(cufftDestroy(fft_plan));
    CUFFT_CHECK(cufftDestroy(ifft_plan));

    return result;
}

HEstimateFResult h_estimate_f_gpu_with_cache(const std::vector<std::complex<float>>& dat255,
                                               const std::vector<std::complex<float>>& pn255,
                                               HEstimateFGPUCache* cache) {
    HEstimateFResult result;
    result.hf.resize(255);
    result.hf_raw.resize(255);

    size_t complexSize = 255 * sizeof(cuFloatComplex);

    std::vector<cuFloatComplex> h_dat255(255);
    std::vector<cuFloatComplex> h_pn255(255);
    for (int i = 0; i < 255; i++) {
        h_dat255[i] = make_cuFloatComplex(dat255[i].real(), dat255[i].imag());
        h_pn255[i] = make_cuFloatComplex(pn255[i].real(), pn255[i].imag());
    }

    CUDA_CHECK(cudaMemcpy(cache->d_dat255, h_dat255.data(), complexSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cache->d_pn255, h_pn255.data(), complexSize, cudaMemcpyHostToDevice));

    cuFloatComplex* d_fft_dat = NULL;
    cuFloatComplex* d_fft_pn = NULL;
    cuFloatComplex* d_div_result = NULL;

    CUDA_CHECK(cudaMalloc(&d_fft_dat, complexSize));
    CUDA_CHECK(cudaMalloc(&d_fft_pn, complexSize));
    CUDA_CHECK(cudaMalloc(&d_div_result, complexSize));

    CUFFT_CHECK(cufftExecC2C(cache->fft_plan, (cufftComplex*)cache->d_dat255, (cufftComplex*)d_fft_dat, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(cache->fft_plan, (cufftComplex*)cache->d_pn255, (cufftComplex*)d_fft_pn, CUFFT_FORWARD));

    int blockSize = 256;
    int gridSize = (255 + blockSize - 1) / blockSize;
    complex_divide_kernel<<<gridSize, blockSize>>>(d_fft_dat, d_fft_pn, d_div_result, 255);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(cache->ifft_plan, (cufftComplex*)d_div_result, (cufftComplex*)cache->d_hf, CUFFT_INVERSE));

    set_last_indices_zero_kernel<<<gridSize, blockSize>>>(cache->d_hf);
    CUDA_CHECK(cudaGetLastError());

    std::vector<cuFloatComplex> h_hf(255);
    CUDA_CHECK(cudaMemcpy(h_hf.data(), cache->d_hf, complexSize, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 255; i++) {
        float scale = 1.0f / 255.0f;
        result.hf_raw[i] = std::complex<float>(cuCrealf(h_hf[i]) * scale, cuCimagf(h_hf[i]) * scale);
    }

    result.hf = result.hf_raw;

    float max_val = 0.0f;
    for (int i = 0; i < 255; i++) {
        float abs_val = std::abs(result.hf[i]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }

    float threshold = 0.03f * max_val;
    for (int i = 0; i < 255; i++) {
        float abs_val = std::abs(result.hf[i]);
        if (abs_val < threshold) {
            result.hf[i] = std::complex<float>(0.0f, 0.0f);
        }
    }

    CUDA_CHECK(cudaFree(d_fft_dat));
    CUDA_CHECK(cudaFree(d_fft_pn));
    CUDA_CHECK(cudaFree(d_div_result));

    return result;
}
