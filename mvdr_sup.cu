// mvdr_sup.cu - 空域杂波抑制的GPU实现 (简化版本)
#include "mvdr_sup.h"
#include <iostream>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void permute_surdata_kernel(
    const cuFloatComplex* __restrict__ d_surdata,
    cuFloatComplex* __restrict__ d_suptmp,
    int nSymbol,
    int nDpl,
    int nsurchan)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nSymbol * nDpl * nsurchan;
    
    if (idx < total) {
        int nSy = idx / (nDpl * nsurchan);
        int rem = idx % (nDpl * nsurchan);
        int nd = rem / nsurchan;
        int ch = rem % nsurchan;
        
        int out_idx = nd * nsurchan * nSymbol + ch * nSymbol + nSy;
        d_suptmp[out_idx] = d_surdata[idx];
    }
}

__global__ void compute_covariance_kernel(
    const cuFloatComplex* __restrict__ d_suptmp,
    cuFloatComplex* __restrict__ d_rxdata,
    int nSymbol,
    int nDpl,
    int nsurchan)
{
    int block = blockIdx.x;
    int thread = threadIdx.x;
    
    if (block < nSymbol && thread < nsurchan * nsurchan) {
        int ch_i = thread / nsurchan;
        int ch_j = thread % nsurchan;
        
        cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
        
        for (int nd = 0; nd < nDpl; nd++) {
            int idx_i = nd * nsurchan * nSymbol + ch_i * nSymbol + block;
            int idx_j = nd * nsurchan * nSymbol + ch_j * nSymbol + block;
            
            cuFloatComplex val_i = d_suptmp[idx_i];
            cuFloatComplex val_j = d_suptmp[idx_j];
            
            float real_part = cuCrealf(val_i) * cuCrealf(val_j) + cuCimagf(val_i) * cuCimagf(val_j);
            float imag_part = cuCimagf(val_i) * cuCrealf(val_j) - cuCrealf(val_i) * cuCimagf(val_j);
            
            sum = make_cuFloatComplex(
                cuCrealf(sum) + real_part,
                cuCimagf(sum) + imag_part
            );
        }
        
        float inv_nDpl = 1.0f / (float)nDpl;
        int out_idx = block * nsurchan * nsurchan + thread;
        d_rxdata[out_idx] = make_cuFloatComplex(
            cuCrealf(sum) * inv_nDpl,
            cuCimagf(sum) * inv_nDpl
        );
    }
}

__device__ __forceinline__ void matrix_invert_2x2(cuFloatComplex* mat) {
    cuFloatComplex a = mat[0];
    cuFloatComplex b = mat[1];
    cuFloatComplex c = mat[2];
    cuFloatComplex d = mat[3];
    
    float det_real = cuCrealf(a) * cuCrealf(d) - cuCimagf(a) * cuCimagf(d) 
                    - cuCrealf(b) * cuCrealf(c) + cuCimagf(b) * cuCimagf(c);
    float det_imag = cuCrealf(a) * cuCimagf(d) + cuCimagf(a) * cuCrealf(d)
                    - cuCrealf(b) * cuCimagf(c) - cuCimagf(b) * cuCrealf(c);
    
    float det_sq = det_real * det_real + det_imag * det_imag;
    float inv_det_sq = 1.0f / det_sq;
    
    float inv_det_real = det_real * inv_det_sq;
    float inv_det_imag = -det_imag * inv_det_sq;
    
    cuFloatComplex inv_a = make_cuFloatComplex(
        cuCrealf(d) * inv_det_real - cuCimagf(d) * inv_det_imag,
        cuCrealf(d) * inv_det_imag + cuCimagf(d) * inv_det_real
    );
    cuFloatComplex inv_b = make_cuFloatComplex(
        -cuCrealf(b) * inv_det_real + cuCimagf(b) * inv_det_imag,
        -cuCrealf(b) * inv_det_imag - cuCimagf(b) * inv_det_real
    );
    cuFloatComplex inv_c = make_cuFloatComplex(
        -cuCrealf(c) * inv_det_real + cuCimagf(c) * inv_det_imag,
        -cuCrealf(c) * inv_det_imag - cuCimagf(c) * inv_det_real
    );
    cuFloatComplex inv_d = make_cuFloatComplex(
        cuCrealf(a) * inv_det_real - cuCimagf(a) * inv_det_imag,
        cuCrealf(a) * inv_det_imag + cuCimagf(a) * inv_det_real
    );
    
    mat[0] = inv_a;
    mat[1] = inv_b;
    mat[2] = inv_c;
    mat[3] = inv_d;
}

__global__ void invert_covariance_2ch_kernel(
    cuFloatComplex* __restrict__ d_rxdata,
    int nSymbol)
{
    int block = blockIdx.x;
    
    if (block < nSymbol) {
        cuFloatComplex mat[4];
        int base = block * 4;
        
        mat[0] = d_rxdata[base + 0];
        mat[1] = d_rxdata[base + 1];
        mat[2] = d_rxdata[base + 2];
        mat[3] = d_rxdata[base + 3];
        
        matrix_invert_2x2(mat);
        
        d_rxdata[base + 0] = mat[0];
        d_rxdata[base + 1] = mat[1];
        d_rxdata[base + 2] = mat[2];
        d_rxdata[base + 3] = mat[3];
    }
}

__global__ void apply_suppression_2ch_kernel(
    const cuFloatComplex* __restrict__ d_suptmp,
    const cuFloatComplex* __restrict__ d_rxdata,
    cuFloatComplex* __restrict__ d_suptmp_out,
    int nSymbol,
    int nDpl)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nDpl * 2 * nSymbol;
    
    if (idx < total) {
        int nd = idx / (2 * nSymbol);
        int ch = (idx % (2 * nSymbol)) / nSymbol;
        int nSy = idx % nSymbol;
        
        int rx_base = nSy * 4;
        cuFloatComplex rx00 = d_rxdata[rx_base + 0];
        cuFloatComplex rx01 = d_rxdata[rx_base + 1];
        cuFloatComplex rx10 = d_rxdata[rx_base + 2];
        cuFloatComplex rx11 = d_rxdata[rx_base + 3];
        
        int suptmp0_idx = nd * 2 * nSymbol + 0 * nSymbol + nSy;
        int suptmp1_idx = nd * 2 * nSymbol + 1 * nSymbol + nSy;
        cuFloatComplex s0 = d_suptmp[suptmp0_idx];
        cuFloatComplex s1 = d_suptmp[suptmp1_idx];
        
        float out0_real = cuCrealf(s0) * cuCrealf(rx00) - cuCimagf(s0) * cuCimagf(rx00)
                         + cuCrealf(s1) * cuCrealf(rx10) - cuCimagf(s1) * cuCimagf(rx10);
        float out0_imag = cuCrealf(s0) * cuCimagf(rx00) + cuCimagf(s0) * cuCrealf(rx00)
                         + cuCrealf(s1) * cuCimagf(rx10) + cuCimagf(s1) * cuCrealf(rx10);
        
        float out1_real = cuCrealf(s0) * cuCrealf(rx01) - cuCimagf(s0) * cuCimagf(rx01)
                         + cuCrealf(s1) * cuCrealf(rx11) - cuCimagf(s1) * cuCimagf(rx11);
        float out1_imag = cuCrealf(s0) * cuCimagf(rx01) + cuCimagf(s0) * cuCrealf(rx01)
                         + cuCrealf(s1) * cuCimagf(rx11) + cuCimagf(s1) * cuCrealf(rx11);
        
        d_suptmp_out[suptmp0_idx] = make_cuFloatComplex(out0_real, out0_imag);
        d_suptmp_out[suptmp1_idx] = make_cuFloatComplex(out1_real, out1_imag);
    }
}

__global__ void permute_back_kernel(
    const cuFloatComplex* __restrict__ d_suptmp,
    cuFloatComplex* __restrict__ d_supdata,
    int nSymbol,
    int nDpl,
    int nsurchan)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nSymbol * nDpl * nsurchan;
    
    if (idx < total) {
        int nSy = idx / (nDpl * nsurchan);
        int rem = idx % (nDpl * nsurchan);
        int nd = rem / nsurchan;
        int ch = rem % nsurchan;
        
        int in_idx = nd * nsurchan * nSymbol + ch * nSymbol + nSy;
        d_supdata[idx] = d_suptmp[in_idx];
    }
}

MVDRSupGPUCache* create_mvdr_sup_gpu_cache(int max_nSymbol, int max_nDpl, int max_nsurchan) {
    MVDRSupGPUCache* cache = (MVDRSupGPUCache*)malloc(sizeof(MVDRSupGPUCache));
    if (!cache) {
        return NULL;
    }
    
    cache->max_nSymbol = max_nSymbol;
    cache->max_nDpl = max_nDpl;
    cache->max_nsurchan = max_nsurchan;
    cache->is_valid = false;
    
    size_t surdata_size = max_nSymbol * max_nDpl * max_nsurchan * sizeof(cuFloatComplex);
    size_t suptmp_size = max_nDpl * max_nsurchan * max_nSymbol * sizeof(cuFloatComplex);
    size_t rxdata_size = max_nSymbol * max_nsurchan * max_nsurchan * sizeof(cuFloatComplex);
    
    CUDA_CHECK(cudaMalloc(&cache->d_surdata, surdata_size));
    CUDA_CHECK(cudaMalloc(&cache->d_suptmp, suptmp_size));
    CUDA_CHECK(cudaMalloc(&cache->d_rxdata, rxdata_size));
    CUDA_CHECK(cudaMalloc(&cache->d_supdata, surdata_size));
    CUDA_CHECK(cudaMalloc(&cache->d_work1, suptmp_size));
    CUDA_CHECK(cudaMalloc(&cache->d_work2, suptmp_size));
    CUDA_CHECK(cudaMalloc(&cache->d_work3, suptmp_size));
    
    cache->is_valid = true;
    return cache;
}

void destroy_mvdr_sup_gpu_cache(MVDRSupGPUCache* cache) {
    if (!cache) {
        return;
    }
    if (cache->is_valid) {
        CUDA_CHECK(cudaFree(cache->d_surdata));
        CUDA_CHECK(cudaFree(cache->d_suptmp));
        CUDA_CHECK(cudaFree(cache->d_rxdata));
        CUDA_CHECK(cudaFree(cache->d_supdata));
        CUDA_CHECK(cudaFree(cache->d_work1));
        CUDA_CHECK(cudaFree(cache->d_work2));
        CUDA_CHECK(cudaFree(cache->d_work3));
    }
    free(cache);
}

std::vector<std::complex<float>> mvdr_sup_gpu_with_cache(
    const std::vector<std::complex<float>>& surdata,
    int nSymbol,
    int nDpl,
    int nsurchan,
    MVDRSupGPUCache* cache)
{
    std::vector<std::complex<float>> result(surdata.size());
    
    if (!cache || !cache->is_valid || 
        nSymbol > cache->max_nSymbol || 
        nDpl > cache->max_nDpl || 
        nsurchan > cache->max_nsurchan) {
        return result;
    }
    
    std::vector<cuFloatComplex> h_surdata(surdata.size());
    for (size_t i = 0; i < surdata.size(); i++) {
        h_surdata[i] = make_cuFloatComplex(surdata[i].real(), surdata[i].imag());
    }
    
    size_t data_size = surdata.size() * sizeof(cuFloatComplex);
    CUDA_CHECK(cudaMemcpy(cache->d_surdata, h_surdata.data(), data_size, cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int gridSize = (surdata.size() + blockSize - 1) / blockSize;
    permute_surdata_kernel<<<gridSize, blockSize>>>(cache->d_surdata, cache->d_suptmp, nSymbol, nDpl, nsurchan);
    CUDA_CHECK(cudaGetLastError());
    
    int covGridSize = nSymbol;
    int covBlockSize = nsurchan * nsurchan;
    compute_covariance_kernel<<<covGridSize, covBlockSize>>>(cache->d_suptmp, cache->d_rxdata, nSymbol, nDpl, nsurchan);
    CUDA_CHECK(cudaGetLastError());
    
    invert_covariance_2ch_kernel<<<nSymbol, 1>>>(cache->d_rxdata, nSymbol);
    CUDA_CHECK(cudaGetLastError());
    
    int applyGridSize = (nDpl * nsurchan * nSymbol + blockSize - 1) / blockSize;
    apply_suppression_2ch_kernel<<<applyGridSize, blockSize>>>(cache->d_suptmp, cache->d_rxdata, cache->d_work1, nSymbol, nDpl);
    CUDA_CHECK(cudaGetLastError());
    
    permute_back_kernel<<<gridSize, blockSize>>>(cache->d_work1, cache->d_supdata, nSymbol, nDpl, nsurchan);
    CUDA_CHECK(cudaGetLastError());
    
    std::vector<cuFloatComplex> h_supdata(surdata.size());
    CUDA_CHECK(cudaMemcpy(h_supdata.data(), cache->d_supdata, data_size, cudaMemcpyDeviceToHost));
    
    for (size_t i = 0; i < surdata.size(); i++) {
        result[i] = std::complex<float>(cuCrealf(h_supdata[i]), cuCimagf(h_supdata[i]));
    }
    
    return result;
}

std::vector<std::complex<float>> mvdr_sup_gpu(
    const std::vector<std::complex<float>>& surdata,
    int nSymbol,
    int nDpl,
    int nsurchan)
{
    MVDRSupGPUCache* cache = create_mvdr_sup_gpu_cache(nSymbol, nDpl, nsurchan);
    if (!cache) {
        return std::vector<std::complex<float>>();
    }
    
    std::vector<std::complex<float>> result = mvdr_sup_gpu_with_cache(surdata, nSymbol, nDpl, nsurchan, cache);
    destroy_mvdr_sup_gpu_cache(cache);
    
    return result;
}