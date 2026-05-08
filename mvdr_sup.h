// mvdr_sup.h - 空域杂波抑制模块
#ifndef MVDR_SUP_H
#define MVDR_SUP_H

#include <complex>
#include <vector>
#include <cuComplex.h>
#include <cuda_runtime.h>

struct MVDRSupGPUCache {
    bool is_valid;
    int max_nSymbol;
    int max_nDpl;
    int max_nsurchan;
    
    cuFloatComplex* d_surdata;
    cuFloatComplex* d_suptmp;
    cuFloatComplex* d_rxdata;
    cuFloatComplex* d_supdata;
    
    cuFloatComplex* d_work1;
    cuFloatComplex* d_work2;
    cuFloatComplex* d_work3;
};

MVDRSupGPUCache* create_mvdr_sup_gpu_cache(int max_nSymbol, int max_nDpl, int max_nsurchan);
void destroy_mvdr_sup_gpu_cache(MVDRSupGPUCache* cache);

std::vector<std::complex<float>> mvdr_sup_gpu(
    const std::vector<std::complex<float>>& surdata,
    int nSymbol,
    int nDpl,
    int nsurchan);

std::vector<std::complex<float>> mvdr_sup_gpu_with_cache(
    const std::vector<std::complex<float>>& surdata,
    int nSymbol,
    int nDpl,
    int nsurchan,
    MVDRSupGPUCache* cache);

#endif // MVDR_SUP_H