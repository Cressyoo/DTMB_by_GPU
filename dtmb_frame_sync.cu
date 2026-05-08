// dtmb_frame_sync.cu
#include "dtmb_frame_sync.h"
#include "dtmb_pn420_data.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cuComplex.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void xcorr_kernel_optimized(const cuFloatComplex* d_x, const cuFloatComplex* d_y,
                                         cuFloatComplex* d_result, int n, int m, int maxlag) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int numThreads = gridDim.x * blockDim.x;
    
    for (int lag = tid; lag <= maxlag; lag += numThreads) {
        int overlapLen = min(m, n - lag);
        cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
        
        for (int i = 0; i < overlapLen; i++) {
            float real_y = cuCrealf(d_y[i]);
            cuFloatComplex x_val = d_x[lag + i];
            
            if (real_y > 0) {
                sum = make_cuFloatComplex(
                    cuCrealf(sum) + cuCrealf(x_val),
                    cuCimagf(sum) + cuCimagf(x_val)
                );
            } else {
                sum = make_cuFloatComplex(
                    cuCrealf(sum) - cuCrealf(x_val),
                    cuCimagf(sum) - cuCimagf(x_val)
                );
            }
        }
        
        d_result[lag] = sum;
    }
}

__global__ void xcorr_kernel(const cuFloatComplex* d_x, const cuFloatComplex* d_y,
                              cuFloatComplex* d_result, int n, int m, int maxlag) {
    int lag = blockIdx.x;
    int threadIdxInBlock = threadIdx.x;
    int blockSize = 64;

    if (lag <= maxlag) {
        int overlapLen = min(m, n - lag);
        cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);

        for (int i = threadIdxInBlock; i < overlapLen; i += blockSize) {
            float real_y = cuCrealf(d_y[i]);
            cuFloatComplex x_val = d_x[lag + i];
            
            if (real_y > 0) {
                sum = make_cuFloatComplex(
                    cuCrealf(sum) + cuCrealf(x_val),
                    cuCimagf(sum) + cuCimagf(x_val)
                );
            } else {
                sum = make_cuFloatComplex(
                    cuCrealf(sum) - cuCrealf(x_val),
                    cuCimagf(sum) - cuCimagf(x_val)
                );
            }
        }

        __shared__ cuFloatComplex block_sum[64];
        block_sum[threadIdxInBlock] = sum;
        __syncthreads();

        for (int s = blockSize / 2; s > 0; s >>= 1) {
            if (threadIdxInBlock < s) {
                block_sum[threadIdxInBlock] = make_cuFloatComplex(
                    cuCrealf(block_sum[threadIdxInBlock]) + cuCrealf(block_sum[threadIdxInBlock + s]),
                    cuCimagf(block_sum[threadIdxInBlock]) + cuCimagf(block_sum[threadIdxInBlock + s])
                );
            }
            __syncthreads();
        }

        if (threadIdxInBlock == 0) {
            d_result[lag] = block_sum[0];
        }
    }
}

std::vector<std::complex<float>> xcorr_gpu(const std::vector<std::complex<float>>& x,
                                          const std::vector<std::complex<float>>& y,
                                          int maxlag) {
    int n = x.size();
    int m = y.size();
    int len = maxlag + 1;
    std::vector<std::complex<float>> result(len, 0);

    cuFloatComplex* d_x = NULL;
    cuFloatComplex* d_y = NULL;
    cuFloatComplex* d_result = NULL;

    size_t xSize = n * sizeof(cuFloatComplex);
    size_t ySize = m * sizeof(cuFloatComplex);
    size_t resultSize = len * sizeof(cuFloatComplex);

    CUDA_CHECK(cudaMalloc(&d_x, xSize));
    CUDA_CHECK(cudaMalloc(&d_y, ySize));
    CUDA_CHECK(cudaMalloc(&d_result, resultSize));

    std::vector<cuFloatComplex> h_x(n);
    std::vector<cuFloatComplex> h_y(m);
    for (int i = 0; i < n; i++) {
        h_x[i] = make_cuFloatComplex(x[i].real(), x[i].imag());
    }
    for (int i = 0; i < m; i++) {
        h_y[i] = make_cuFloatComplex(y[i].real(), y[i].imag());
    }

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), xSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), ySize, cudaMemcpyHostToDevice));

    int blocksPerLag = 1;
    int threadsPerBlock = 64;
    xcorr_kernel<<<len, threadsPerBlock>>>(d_x, d_y, d_result, n, m, maxlag);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<cuFloatComplex> h_result(len);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, resultSize, cudaMemcpyDeviceToHost));

    for (int i = 0; i < len; i++) {
        result[i] = std::complex<float>(cuCrealf(h_result[i]), cuCimagf(h_result[i]));
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_result));

    return result;
}

std::vector<std::complex<float>> xcorr(const std::vector<std::complex<float>>& x,
                                      const std::vector<std::complex<float>>& y,
                                      int maxlag) {
    int n = x.size();
    int m = y.size();
    int len = maxlag + 1;
    std::vector<std::complex<float>> result(len, 0);

    for (int lag = 0; lag <= maxlag; lag++) {
        std::complex<float> sum = 0;

        int overlapLen = std::min(m, n - lag);
        if (overlapLen <= 0) break;

        for (int i = 0; i < overlapLen; i++) {
            float real_y = std::real(y[i]);
            if (real_y > 0) {
                sum += x[lag + i];
            } else {
                sum -= x[lag + i];
            }
        }

        result[lag] = sum;
    }

    return result;
}

void cal_syn(int in0, int in1, int& frame_bias, int& frame_syn) {       
    int flag = 0;
    int delta = in1 - in0 - 4200;

    if (delta < 0) {
        flag = 1;
        delta = -delta;
    }

    if (delta > 112) {
        std::cerr << "Error: Out of sync range" << std::endl;
        frame_bias = -1;
        frame_syn = -1;
        return;
    }

    int v = delta + flag;
    if (v % 2 == 1) {
        frame_syn = 225 - delta;
    } else {
        frame_syn = delta;
    }

    if (flag == 1) {
        frame_bias = delta / 2;
    } else {
        frame_bias = - (delta + 1) / 2;
    }

    frame_bias = in0 - frame_bias - 1;

    if (frame_bias > 4200) {
        int a0 = frame_bias / 4200;
        frame_bias = frame_bias % 4200;
        frame_syn = frame_syn - a0;
    }

    if (frame_syn <= 0) {
        frame_syn = frame_syn + 225;
    }

    if (frame_bias <= 0) {
        frame_bias = frame_bias + 4200;
        frame_syn = frame_syn + 1;
        if (frame_syn == 226) {
            frame_syn = 1;
        }
    }
}

void dtmb_syn(const std::vector<std::complex<float>>& refdata,  
              const std::vector<std::complex<float>>& pn420,    
              int frame_bias_pre,
              int frame_syn_pre,
              int& frame_bias,
              int& frame_syn) {
    int avgdata = 4200;

    PeakResult peakResult = dtmb_syn_gpu(refdata, pn420);
    
    int ind0 = peakResult.ind0;
    float val0 = peakResult.val0;
    int ind1 = peakResult.ind1;
    float val1 = peakResult.val1;

    std::cout << "dtmb_syn: ind0 = " << ind0 << ", max_val0 = " << val0 << std::endl;
    std::cout << "dtmb_syn: ind1 = " << ind1 << ", max_val1 = " << val1 << std::endl;
    std::cout << "dtmb_syn: delta = " << (ind1 - ind0 - avgdata) << std::endl;

    int b0, s0;
    cal_syn(ind0, ind1, b0, s0);

    frame_bias = b0;
    frame_syn = s0;

    std::cout << "dtmb_syn result: frame_bias = " << frame_bias << ", frame_syn = " << frame_syn << std::endl;
}

void dtmb_syn_with_cache(const std::vector<std::complex<float>>& refdata,  
                        const std::vector<std::complex<float>>& pn420,    
                        int frame_bias_pre,
                        int frame_syn_pre,
                        int& frame_bias,
                        int& frame_syn,
                        DTMBGPUCache* cache) {
    int avgdata = 4200;

    PeakResult peakResult = dtmb_syn_gpu_with_cache(refdata, pn420, cache);
    
    int ind0 = peakResult.ind0;
    float val0 = peakResult.val0;
    int ind1 = peakResult.ind1;
    float val1 = peakResult.val1;

    std::cout << "dtmb_syn: ind0 = " << ind0 << ", max_val0 = " << val0 << std::endl;
    std::cout << "dtmb_syn: ind1 = " << ind1 << ", max_val1 = " << val1 << std::endl;
    std::cout << "dtmb_syn: delta = " << (ind1 - ind0 - avgdata) << std::endl;

    int b0, s0;
    cal_syn(ind0, ind1, b0, s0);

    frame_bias = b0;
    frame_syn = s0;

    std::cout << "dtmb_syn result: frame_bias = " << frame_bias << ", frame_syn = " << frame_syn << std::endl;
}

bool read_pn420(const std::string& filename, std::vector<std::complex<float>>& pn420) {
    pn420.resize(420);

    for (int i = 0; i < 420; i++) {
        float real = pn420_0[i];
        pn420[i] = std::complex<float>(real, 0.0f);
    }

    return true;
}

__global__ void frame_sync_compute_abs_kernel(const cuFloatComplex* d_xcorr, float* d_abs, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        cuFloatComplex val = d_xcorr[idx];
        float real = cuCrealf(val);
        float imag = cuCimagf(val);
        d_abs[idx] = sqrt(real * real + imag * imag);
    }
}

__global__ void find_peak_kernel(const float* d_abs, int len, int start, int end, int* out_ind, float* out_val) {
    __shared__ float sh_val[64];
    __shared__ int sh_ind[64];
    
    int thread_idx = threadIdx.x;
    
    sh_val[thread_idx] = -1.0f;
    sh_ind[thread_idx] = -1;
    __syncthreads();
    
    for (int idx = start + thread_idx; idx < end; idx += 64) {
        float val = d_abs[idx];
        if (val > sh_val[thread_idx]) {
            sh_val[thread_idx] = val;
            sh_ind[thread_idx] = idx;
        }
    }
    __syncthreads();
    
    for (int s = 32; s > 0; s >>= 1) {
        if (thread_idx < s) {
            if (sh_val[thread_idx] < sh_val[thread_idx + s]) {
                sh_val[thread_idx] = sh_val[thread_idx + s];
                sh_ind[thread_idx] = sh_ind[thread_idx + s];
            }
        }
        __syncthreads();
    }
    
    if (thread_idx == 0) {
        *out_ind = sh_ind[0];
        *out_val = sh_val[0];
    }
}

DTMBGPUCache* create_dtmb_gpu_cache(int max_refdata_size, int maxlag) {
    DTMBGPUCache* cache = (DTMBGPUCache*)malloc(sizeof(DTMBGPUCache));
    if (!cache) {
        return NULL;
    }
    
    cache->max_refdata_size = max_refdata_size;
    cache->maxlag = maxlag;
    cache->is_valid = false;
    
    int len = maxlag + 1;
    
    size_t xSize = max_refdata_size * sizeof(cuFloatComplex);
    size_t ySize = 420 * sizeof(cuFloatComplex);
    size_t xcorrSize = len * sizeof(cuFloatComplex);
    size_t absSize = len * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&cache->d_x, xSize));
    CUDA_CHECK(cudaMalloc(&cache->d_y, ySize));
    CUDA_CHECK(cudaMalloc(&cache->d_xcorr, xcorrSize));
    CUDA_CHECK(cudaMalloc(&cache->d_abs, absSize));
    CUDA_CHECK(cudaMalloc(&cache->d_ind0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cache->d_val0, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cache->d_ind1, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cache->d_val1, sizeof(float)));
    
    cache->is_valid = true;
    return cache;
}

void destroy_dtmb_gpu_cache(DTMBGPUCache* cache) {
    if (!cache) {
        return;
    }
    if (cache->is_valid) {
        CUDA_CHECK(cudaFree(cache->d_x));
        CUDA_CHECK(cudaFree(cache->d_y));
        CUDA_CHECK(cudaFree(cache->d_xcorr));
        CUDA_CHECK(cudaFree(cache->d_abs));
        CUDA_CHECK(cudaFree(cache->d_ind0));
        CUDA_CHECK(cudaFree(cache->d_val0));
        CUDA_CHECK(cudaFree(cache->d_ind1));
        CUDA_CHECK(cudaFree(cache->d_val1));
    }
    free(cache);
}

PeakResult dtmb_syn_gpu_with_cache(const std::vector<std::complex<float>>& refdata,
                                     const std::vector<std::complex<float>>& pn420,
                                     DTMBGPUCache* cache) {
    PeakResult result;
    int maxlag = cache->maxlag;
    int len = maxlag + 1;
    
    int n = refdata.size();
    int m = pn420.size();
    
    size_t xSize = n * sizeof(cuFloatComplex);
    size_t ySize = m * sizeof(cuFloatComplex);
    
    std::vector<cuFloatComplex> h_x(n);
    for (int i = 0; i < n; i++) {
        h_x[i] = make_cuFloatComplex(refdata[i].real(), refdata[i].imag());
    }
    
    CUDA_CHECK(cudaMemcpy(cache->d_x, h_x.data(), xSize, cudaMemcpyHostToDevice));
    
    static bool pn420_copied = false;
    if (!pn420_copied) {
        std::vector<cuFloatComplex> h_y(m);
        for (int i = 0; i < m; i++) {
            h_y[i] = make_cuFloatComplex(pn420[i].real(), pn420[i].imag());
        }
        CUDA_CHECK(cudaMemcpy(cache->d_y, h_y.data(), ySize, cudaMemcpyHostToDevice));
        pn420_copied = true;
    }
    
    int threadsPerBlock = 256;
    int blocksPerGrid = 64;
    xcorr_kernel_optimized<<<blocksPerGrid, threadsPerBlock>>>(cache->d_x, cache->d_y, cache->d_xcorr, n, m, maxlag);
    CUDA_CHECK(cudaGetLastError());
    
    int absBlockSize = 256;
    int absGridSize = (len + absBlockSize - 1) / absBlockSize;
    frame_sync_compute_abs_kernel<<<absGridSize, absBlockSize>>>(cache->d_xcorr, cache->d_abs, len);
    CUDA_CHECK(cudaGetLastError());
    
    int searchLen0 = min(512 * 9, len);
    find_peak_kernel<<<1, 64>>>(cache->d_abs, len, 0, searchLen0, cache->d_ind0, cache->d_val0);
    CUDA_CHECK(cudaGetLastError());
    
    int h_ind0, h_ind1;
    float h_val0, h_val1;
    CUDA_CHECK(cudaMemcpy(&h_ind0, cache->d_ind0, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_val0, cache->d_val0, sizeof(float), cudaMemcpyDeviceToHost));
    
    int avgdata = 4200;
    int offdata = 128;
    int ind1_start = h_ind0 + avgdata - offdata;
    int ind1_end = h_ind0 + avgdata + offdata;
    if (ind1_start < 0) ind1_start = 0;
    if (ind1_end >= len) ind1_end = len;
    
    find_peak_kernel<<<1, 64>>>(cache->d_abs, len, ind1_start, ind1_end, cache->d_ind1, cache->d_val1);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(&h_ind1, cache->d_ind1, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_val1, cache->d_val1, sizeof(float), cudaMemcpyDeviceToHost));
    
    result.ind0 = h_ind0;
    result.val0 = h_val0;
    result.ind1 = h_ind1;
    result.val1 = h_val1;
    
    return result;
}

PeakResult dtmb_syn_gpu(const std::vector<std::complex<float>>& refdata,
                         const std::vector<std::complex<float>>& pn420) {
    PeakResult result;
    int maxlag = 3 * 4200;
    int len = maxlag + 1;
    
    cuFloatComplex* d_x = NULL;
    cuFloatComplex* d_y = NULL;
    cuFloatComplex* d_xcorr = NULL;
    float* d_abs = NULL;
    int* d_ind0 = NULL;
    float* d_val0 = NULL;
    int* d_ind1 = NULL;
    float* d_val1 = NULL;
    
    int n = refdata.size();
    int m = pn420.size();
    
    size_t xSize = n * sizeof(cuFloatComplex);
    size_t ySize = m * sizeof(cuFloatComplex);
    size_t xcorrSize = len * sizeof(cuFloatComplex);
    size_t absSize = len * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_x, xSize));
    CUDA_CHECK(cudaMalloc(&d_y, ySize));
    CUDA_CHECK(cudaMalloc(&d_xcorr, xcorrSize));
    CUDA_CHECK(cudaMalloc(&d_abs, absSize));
    CUDA_CHECK(cudaMalloc(&d_ind0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val0, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ind1, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val1, sizeof(float)));
    
    std::vector<cuFloatComplex> h_x(n);
    std::vector<cuFloatComplex> h_y(m);
    for (int i = 0; i < n; i++) {
        h_x[i] = make_cuFloatComplex(refdata[i].real(), refdata[i].imag());
    }
    for (int i = 0; i < m; i++) {
        h_y[i] = make_cuFloatComplex(pn420[i].real(), pn420[i].imag());
    }
    
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), xSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), ySize, cudaMemcpyHostToDevice));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = 64;
    xcorr_kernel_optimized<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_xcorr, n, m, maxlag);
    CUDA_CHECK(cudaGetLastError());
    
    int absBlockSize = 256;
    int absGridSize = (len + absBlockSize - 1) / absBlockSize;
    frame_sync_compute_abs_kernel<<<absGridSize, absBlockSize>>>(d_xcorr, d_abs, len);
    CUDA_CHECK(cudaGetLastError());
    
    int searchLen0 = min(512 * 9, len);
    find_peak_kernel<<<1, 64>>>(d_abs, len, 0, searchLen0, d_ind0, d_val0);
    CUDA_CHECK(cudaGetLastError());
    
    int h_ind0, h_ind1;
    float h_val0, h_val1;
    CUDA_CHECK(cudaMemcpy(&h_ind0, d_ind0, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_val0, d_val0, sizeof(float), cudaMemcpyDeviceToHost));
    
    int avgdata = 4200;
    int offdata = 128;
    int ind1_start = h_ind0 + avgdata - offdata;
    int ind1_end = h_ind0 + avgdata + offdata;
    if (ind1_start < 0) ind1_start = 0;
    if (ind1_end >= len) ind1_end = len;
    
    find_peak_kernel<<<1, 64>>>(d_abs, len, ind1_start, ind1_end, d_ind1, d_val1);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(&h_ind1, d_ind1, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_val1, d_val1, sizeof(float), cudaMemcpyDeviceToHost));
    
    result.ind0 = h_ind0;
    result.val0 = h_val0;
    result.ind1 = h_ind1;
    result.val1 = h_val1;
    
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_xcorr));
    CUDA_CHECK(cudaFree(d_abs));
    CUDA_CHECK(cudaFree(d_ind0));
    CUDA_CHECK(cudaFree(d_val0));
    CUDA_CHECK(cudaFree(d_ind1));
    CUDA_CHECK(cudaFree(d_val1));
    
    return result;
}

PeakResult dtmb_syn_gpu_device_with_cache(const cuFloatComplex* d_refdata,
                                            int refdata_size,
                                            const std::vector<std::complex<float>>& pn420,
                                            DTMBGPUCache* cache) {
    PeakResult result;
    int maxlag = cache->maxlag;
    int len = maxlag + 1;
    
    int n = refdata_size;
    int m = pn420.size();
    
    static bool pn420_copied = false;
    if (!pn420_copied) {
        std::vector<cuFloatComplex> h_y(m);
        for (int i = 0; i < m; i++) {
            h_y[i] = make_cuFloatComplex(pn420[i].real(), pn420[i].imag());
        }
        CUDA_CHECK(cudaMemcpy(cache->d_y, h_y.data(), m * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
        pn420_copied = true;
    }
    
    int threadsPerBlock = 256;
    int blocksPerGrid = 64;
    xcorr_kernel_optimized<<<blocksPerGrid, threadsPerBlock>>>(
        d_refdata, cache->d_y, cache->d_xcorr, n, m, maxlag);
    CUDA_CHECK(cudaGetLastError());
    
    int absBlockSize = 256;
    int absGridSize = (len + absBlockSize - 1) / absBlockSize;
    frame_sync_compute_abs_kernel<<<absGridSize, absBlockSize>>>(cache->d_xcorr, cache->d_abs, len);
    CUDA_CHECK(cudaGetLastError());
    
    int searchLen0 = min(512 * 9, len);
    find_peak_kernel<<<1, 64>>>(cache->d_abs, len, 0, searchLen0, cache->d_ind0, cache->d_val0);
    CUDA_CHECK(cudaGetLastError());
    
    int h_ind0, h_ind1;
    float h_val0, h_val1;
    CUDA_CHECK(cudaMemcpy(&h_ind0, cache->d_ind0, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_val0, cache->d_val0, sizeof(float), cudaMemcpyDeviceToHost));
    
    int avgdata = 4200;
    int offdata = 128;
    int ind1_start = h_ind0 + avgdata - offdata;
    int ind1_end = h_ind0 + avgdata + offdata;
    if (ind1_start < 0) ind1_start = 0;
    if (ind1_end >= len) ind1_end = len;
    
    find_peak_kernel<<<1, 64>>>(cache->d_abs, len, ind1_start, ind1_end, cache->d_ind1, cache->d_val1);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(&h_ind1, cache->d_ind1, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_val1, cache->d_val1, sizeof(float), cudaMemcpyDeviceToHost));
    
    result.ind0 = h_ind0;
    result.val0 = h_val0;
    result.ind1 = h_ind1;
    result.val1 = h_val1;
    
    return result;
}

void dtmb_syn_device_with_cache(const cuFloatComplex* d_refdata,
                                 int refdata_size,
                                 const std::vector<std::complex<float>>& pn420,    
                                 int frame_bias_pre,
                                 int frame_syn_pre,
                                 int& frame_bias,
                                 int& frame_syn,
                                 DTMBGPUCache* cache) {
    int avgdata = 4200;

    PeakResult peakResult = dtmb_syn_gpu_device_with_cache(d_refdata, refdata_size, pn420, cache);
    
    int ind0 = peakResult.ind0;
    float val0 = peakResult.val0;
    int ind1 = peakResult.ind1;
    float val1 = peakResult.val1;

    std::cout << "dtmb_syn: ind0 = " << ind0 << ", max_val0 = " << val0 << std::endl;
    std::cout << "dtmb_syn: ind1 = " << ind1 << ", max_val1 = " << val1 << std::endl;
    std::cout << "dtmb_syn: delta = " << (ind1 - ind0 - avgdata) << std::endl;

    int b0, s0;
    cal_syn(ind0, ind1, b0, s0);

    frame_bias = b0;
    frame_syn = s0;

    std::cout << "dtmb_syn result: frame_bias = " << frame_bias << ", frame_syn = " << frame_syn << std::endl;
}
