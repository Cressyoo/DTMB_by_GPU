#ifndef H_ESTIMATE_IMPROVED_H
#define H_ESTIMATE_IMPROVED_H

#include <complex>
#include <vector>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>

struct HEstimateImprovedResult {
    std::vector<std::complex<float>> hr;
    std::vector<std::complex<float>> hr_raw;
};

struct HEstimateImprovedGPUCache {
    bool is_valid;
    int max_len;
    int max_frame_num;

    cuFloatComplex* d_dat255_batch;
    cuFloatComplex* d_pn255_batch;
    cuFloatComplex* d_hr_batch;
    cuFloatComplex* d_fft_dat_batch;
    cuFloatComplex* d_fft_pn_batch;
    cuFloatComplex* d_conj_mul_result_batch;

    cufftHandle fft_plan_batch;
    cufftHandle ifft_plan_batch;
};

HEstimateImprovedGPUCache* create_h_estimate_improved_gpu_cache_with_batch(int max_frame_num);
void destroy_h_estimate_improved_gpu_cache(HEstimateImprovedGPUCache* cache);

void h_estimate_improved_gpu_batch_with_cache(const std::vector<cuFloatComplex>& dat255_batch,
                                                const std::vector<cuFloatComplex>& pn255_batch,
                                                int frame_num,
                                                std::vector<cuFloatComplex>& hr_batch,
                                                HEstimateImprovedGPUCache* cache);

void h_estimate_improved_gpu_batch_device_with_cache(
                                               int frame_num,
                                               std::vector<cuFloatComplex>& hr_batch,
                                               HEstimateImprovedGPUCache* cache);

#endif // H_ESTIMATE_IMPROVED_H
