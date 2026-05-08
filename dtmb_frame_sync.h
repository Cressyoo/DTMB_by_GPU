// dtmb_frame_sync.h
#ifndef DTMB_FRAME_SYNC_H
#define DTMB_FRAME_SYNC_H

#include <complex>
#include <vector>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>

struct PeakResult {
    int ind0;
    float val0;
    int ind1;
    float val1;
};

struct DTMBGPUCache {
    bool is_valid;
    int max_refdata_size;
    int maxlag;
    
    cuFloatComplex* d_x;
    cuFloatComplex* d_y;
    cuFloatComplex* d_xcorr;
    float* d_abs;
    int* d_ind0;
    float* d_val0;
    int* d_ind1;
    float* d_val1;
};

DTMBGPUCache* create_dtmb_gpu_cache(int max_refdata_size, int maxlag);
void destroy_dtmb_gpu_cache(DTMBGPUCache* cache);

std::vector<std::complex<float>> xcorr_gpu(const std::vector<std::complex<float>>& x,
                                          const std::vector<std::complex<float>>& y,
                                          int maxlag);

PeakResult dtmb_syn_gpu(const std::vector<std::complex<float>>& refdata,
                         const std::vector<std::complex<float>>& pn420);
PeakResult dtmb_syn_gpu_with_cache(const std::vector<std::complex<float>>& refdata,
                                     const std::vector<std::complex<float>>& pn420,
                                     DTMBGPUCache* cache);

PeakResult dtmb_syn_gpu_device_with_cache(const cuFloatComplex* d_refdata,
                                            int refdata_size,
                                            const std::vector<std::complex<float>>& pn420,
                                            DTMBGPUCache* cache);
                                     
void dtmb_syn_with_cache(const std::vector<std::complex<float>>& refdata,  
                        const std::vector<std::complex<float>>& pn420,    
                        int frame_bias_pre,
                        int frame_syn_pre,
                        int& frame_bias,
                        int& frame_syn,
                        DTMBGPUCache* cache);

void dtmb_syn_device_with_cache(const cuFloatComplex* d_refdata,
                                 int refdata_size,
                                 const std::vector<std::complex<float>>& pn420,    
                                 int frame_bias_pre,
                                 int frame_syn_pre,
                                 int& frame_bias,
                                 int& frame_syn,
                                 DTMBGPUCache* cache);

/**
 * @brief DTMB????????
 *
 * @param refdata ????????
 * @param pn420 PN420????
 * @param frame_bias_pre ??????????
 * @param frame_syn_pre ??????????????
 * @param frame_bias ?????????????
 * @param frame_syn ?????????????????
 */
void dtmb_syn(const std::vector<std::complex<float>>& refdata, 
              const std::vector<std::complex<float>>& pn420, 
              int frame_bias_pre, 
              int frame_syn_pre, 
              int& frame_bias, 
              int& frame_syn);

/**
 * @brief ????????
 *
 * @param in0 ????????????
 * @param in1 ????????????
 * @param frame_bias ???????
 * @param frame_syn ???????????
 */
void cal_syn(int in0, int in1, int& frame_bias, int& frame_syn);

/**
 * @brief ???PN420????
 *
 * @param filename PN420???????????
 * @param pn420 ???PN420????
 * @return bool ????????
 */
bool read_pn420(const std::string& filename, std::vector<std::complex<float>>& pn420);

#endif // DTMB_FRAME_SYNC_H
