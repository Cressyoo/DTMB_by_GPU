// h_estimate_f.h - Frequency domain channel estimation
#ifndef H_ESTIMATE_F_H
#define H_ESTIMATE_F_H

#include <complex>
#include <vector>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>

struct HEstimateFResult {
    std::vector<std::complex<float>> hf;
    std::vector<std::complex<float>> hf_raw;
};

struct HEstimateFGPUCache {
    bool is_valid;
    int max_len;
    int max_frame_num;
    
    cuFloatComplex* d_dat255;
    cuFloatComplex* d_pn255;
    cuFloatComplex* d_hf;
    float* d_abs_hf;
    
    cufftHandle fft_plan;
    cufftHandle ifft_plan;
    
    cuFloatComplex* d_dat255_batch;
    cuFloatComplex* d_pn255_batch;
    cuFloatComplex* d_hf_batch;
    cuFloatComplex* d_fft_dat_batch;
    cuFloatComplex* d_fft_pn_batch;
    cuFloatComplex* d_div_result_batch;
    
    cufftHandle fft_plan_batch;
    cufftHandle ifft_plan_batch;
};

HEstimateFGPUCache* create_h_estimate_f_gpu_cache();
HEstimateFGPUCache* create_h_estimate_f_gpu_cache_with_batch(int max_frame_num);
void destroy_h_estimate_f_gpu_cache(HEstimateFGPUCache* cache);

HEstimateFResult h_estimate_f_gpu(const std::vector<std::complex<float>>& dat255,
                                   const std::vector<std::complex<float>>& pn255);

HEstimateFResult h_estimate_f_gpu_with_cache(const std::vector<std::complex<float>>& dat255,
                                               const std::vector<std::complex<float>>& pn255,
                                               HEstimateFGPUCache* cache);

void h_estimate_f_gpu_batch_with_cache(const std::vector<cuFloatComplex>& dat255_batch,
                                        const std::vector<cuFloatComplex>& pn255_batch,
                                        int frame_num,
                                        std::vector<cuFloatComplex>& hf_batch,
                                        HEstimateFGPUCache* cache);

#endif // H_ESTIMATE_F_H
