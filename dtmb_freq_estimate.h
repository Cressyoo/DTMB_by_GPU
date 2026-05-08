// dtmb_freq_estimate.h
#ifndef DTMB_FREQ_ESTIMATE_H
#define DTMB_FREQ_ESTIMATE_H

#include <complex>
#include <vector>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>

struct FreqEstimateResult {
    float freq_delta;           
    float freq_delta_std;      
    float freq_delta_max;      
    float freq_delta_min;      
    std::vector<float> freq_delta_v; 
    
    float freq_delta1;          
    float freq_delta1_std;     
    float freq_delta1_max;     
    float freq_delta1_min;     
    std::vector<float> freq_delta_v1; 
};

struct FreqEstimateGPUCache {
    bool is_valid;
    int max_frame_count;
    
    cuFloatComplex* d_data;
    cuFloatComplex* d_pn420_bz;
    cuFloatComplex* d_z1;
    cuFloatComplex* d_z2;
    float* d_freq_delta_v;
    float* d_freq_delta_v1;
};

FreqEstimateGPUCache* create_freq_estimate_gpu_cache(int max_frame_count);
void destroy_freq_estimate_gpu_cache(FreqEstimateGPUCache* cache);

int get_sync_type(int frame_n);

void get_pn_sequence_params(int syn_b, int& sb0, int& sb1, int& len);

FreqEstimateResult dtmb_freq_estimate(
    const std::vector<std::complex<float>>& data,
    int frame_syn,
    int cnt
);

FreqEstimateResult dtmb_freq_estimate_gpu(
    const std::vector<std::complex<float>>& data,
    int frame_syn,
    int cnt,
    FreqEstimateGPUCache* cache
);

FreqEstimateResult dtmb_freq_estimate_gpu_device(
    const cufftComplex* d_data,
    int data_size,
    int frame_syn,
    int cnt,
    FreqEstimateGPUCache* cache
);

void init_pn420_bz_complex();

// GPU频率补偿相关结构体和函数
struct FreqCompensateGPUCache {
    bool is_valid;
    int max_len;
    cuFloatComplex* d_input;
    cuFloatComplex* d_output;
};

FreqCompensateGPUCache* create_freq_compensate_gpu_cache(int max_len);
void destroy_freq_compensate_gpu_cache(FreqCompensateGPUCache* cache);

// GPU频率补偿 - 返回GPU处理后的数据
std::vector<std::complex<float>> freq_compensate_gpu(
    const std::vector<std::complex<float>>& data,
    float freq_delta,
    float sample_rate,
    FreqCompensateGPUCache* cache);

// GPU频率补偿 - 原地处理（输入输出使用同一缓冲区）
void freq_compensate_gpu_inplace(
    cuFloatComplex* d_data,
    float freq_delta,
    float sample_rate,
    int n);

// GPU双通道频率补偿 - 分别对两个通道进行补偿
std::vector<std::complex<float>> freq_compensate_gpu_dual_channel(
    const std::vector<std::complex<float>>& data,
    float freq_delta,
    float sample_rate,
    FreqCompensateGPUCache* cache);

// 直接在GPU内存上进行双通道频率补偿（原地修改）
void freq_compensate_gpu_dual_channel_inplace(
    cufftComplex* d_data,
    float freq_delta,
    float sample_rate,
    int samples_per_channel);

// 直接在GPU内存上进行多通道频率补偿（原地修改）
void freq_compensate_gpu_multi_channel_inplace(
    cufftComplex* d_data,
    float freq_delta,
    float sample_rate,
    int num_channels,
    int samples_per_channel);

// 从Host内存的cufftComplex数据开始，高效进行双通道频偏补偿
std::vector<std::complex<float>> freq_compensate_gpu_dual_channel_from_cufft(
    cufftComplex* h_data,
    int total_samples,
    float freq_delta,
    float sample_rate,
    FreqCompensateGPUCache* cache);

#endif // DTMB_FREQ_ESTIMATE_H
