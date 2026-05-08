#ifndef REF_CLEAR_H
#define REF_CLEAR_H

#include "h_estimate_improved.h"
#include <complex>
#include <vector>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>

struct RefClearResult {
    std::vector<std::complex<float>> refclrdata;
    std::vector<std::complex<float>> data_in_fft_freq;
    int delta;
};

struct RefClearResultDevice {
    cuFloatComplex* d_refclrdata;
    cuFloatComplex* d_data_in_fft_freq;
    int frame_num;
    int delta;
};

struct RefClearGPUCache {
    bool is_valid;
    int max_frame_num;

    cuFloatComplex* d_dat;
    cuFloatComplex* d_pn420_bz;
    cuFloatComplex* d_refclrdata;

    HEstimateImprovedGPUCache* h_est_improved_cache;

    cufftHandle fft_plan_3780_batch;
    cufftHandle ifft_plan_3780_batch;

    cufftHandle fft_plan_conv_batch;
    cufftHandle ifft_plan_conv_batch;
    cuFloatComplex* d_conv_pn420_padded_batch;
    cuFloatComplex* d_conv_hf_padded_batch;
    cuFloatComplex* d_conv_pn420_fft_batch;
    cuFloatComplex* d_conv_hf_fft_batch;
    cuFloatComplex* d_conv_result_batch;

    cuFloatComplex* d_g1_batch;
    cuFloatComplex* d_g2_batch;
    cuFloatComplex* d_hop_freq_batch;
    cuFloatComplex* d_hop_freq_padded_batch;
    cuFloatComplex* d_data_jun_batch;
    cuFloatComplex* d_data_in_fft_freq_batch;
    cuFloatComplex* d_tmp_batch;

    cuFloatComplex* d_h_fft_freq;

    int* d_frame;
    
    bool pn420_initialized; // 标记pn420_bz是否已初始化
};

RefClearGPUCache* create_ref_clear_gpu_cache(int max_frame_num, HEstimateImprovedGPUCache* external_h_est_improved_cache = nullptr);
void destroy_ref_clear_gpu_cache(RefClearGPUCache* cache, bool free_h_est_improved_cache = true);

RefClearResult ref_clear_gpu_batch(const std::vector<std::complex<float>>& dat,
                                     int frame_syn,
                                     int frame_num,
                                     const std::vector<std::vector<std::complex<float>>>& pn420_bz,
                                     RefClearGPUCache* cache);

RefClearResultDevice ref_clear_gpu_batch_device(
                                    const cuFloatComplex* d_dat_src,
                                    int dat_offset,
                                    int dat_total_size,
                                    int frame_syn,
                                    int frame_num,
                                    const std::vector<std::vector<std::complex<float>>>& pn420_bz,
                                    RefClearGPUCache* cache);

void export_constellation_data_binary(const std::vector<std::complex<float>>& data_in_fft_freq,
                                        const std::string& filename_freq);

void generate_constellation_plot_script_binary(const std::string& filename_freq,
                                                 const std::string& script_filename);

std::vector<std::vector<std::complex<float>>> get_pn420_bz();

#endif // REF_CLEAR_H
