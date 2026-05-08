#include "ref_clear.h"
#include "h_estimate_improved.h"
#include "dtmb_pn420_01_data.h"
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

__global__ void complex_divide_batch_kernel(
    const cuFloatComplex* __restrict__ d_a_batch,
    const cuFloatComplex* __restrict__ d_b_batch,
    cuFloatComplex* __restrict__ d_out_batch,
    int frame_num,
    int frame_size,
    float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = frame_num * frame_size;
    if (idx < total_elements) {
        int fi = idx / frame_size;
        int i = idx % frame_size;

        cuFloatComplex a = d_a_batch[fi * frame_size + i];
        cuFloatComplex b = d_b_batch[fi * frame_size + i];

        float denominator = cuCrealf(b) * cuCrealf(b) + cuCimagf(b) * cuCimagf(b);
        if (denominator > 1e-10f) {
            float real = (cuCrealf(a) * cuCrealf(b) + cuCimagf(a) * cuCimagf(b)) / denominator;
            float imag = (cuCimagf(a) * cuCrealf(b) - cuCrealf(a) * cuCimagf(b)) / denominator;
            d_out_batch[idx] = make_cuFloatComplex(real * scale, imag * scale);
        } else {
            d_out_batch[idx] = make_cuFloatComplex(0.0f, 0.0f);
        }
    }
}

__global__ void normalize_ifft_batch_kernel(cuFloatComplex* __restrict__ d_data, int frame_num, int frame_size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = frame_num * frame_size;
    if (idx < total_elements) {
        d_data[idx] = make_cuFloatComplex(
            cuCrealf(d_data[idx]) * scale,
            cuCimagf(d_data[idx]) * scale
        );
    }
}

__global__ void pad_arrays_for_conv_batch_kernel(
    const cuFloatComplex* __restrict__ d_pn420_batch,
    const cuFloatComplex* __restrict__ d_hf_batch,
    cuFloatComplex* __restrict__ d_pn420_padded_batch,
    cuFloatComplex* __restrict__ d_hf_padded_batch,
    int pn_len,
    int hf_len,
    int conv_len,
    int frame_num
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = frame_num * conv_len;
    if (idx < total_elements) {
        int fi = idx / conv_len;
        int i = idx % conv_len;

        if (i < pn_len) {
            d_pn420_padded_batch[idx] = d_pn420_batch[fi * pn_len + i];
        } else {
            d_pn420_padded_batch[idx] = make_cuFloatComplex(0.0f, 0.0f);
        }

        if (i < hf_len) {
            d_hf_padded_batch[idx] = d_hf_batch[fi * hf_len + i];
        } else {
            d_hf_padded_batch[idx] = make_cuFloatComplex(0.0f, 0.0f);
        }
    }
}

__global__ void complex_multiply_batch_kernel(
    const cuFloatComplex* __restrict__ d_a,
    const cuFloatComplex* __restrict__ d_b,
    cuFloatComplex* __restrict__ d_out,
    int conv_len,
    int frame_num
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = frame_num * conv_len;
    if (idx < total_elements) {
        cuFloatComplex a = d_a[idx];
        cuFloatComplex b = d_b[idx];

        float real_part = cuCrealf(a) * cuCrealf(b) - cuCimagf(a) * cuCimagf(b);
        float imag_part = cuCrealf(a) * cuCimagf(b) + cuCimagf(a) * cuCrealf(b);
        d_out[idx] = make_cuFloatComplex(real_part, imag_part);
    }
}

__global__ void normalize_ifft_conv_batch_kernel(
    cuFloatComplex* d_data,
    int conv_len,
    int frame_num,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = frame_num * conv_len;
    if (idx < total_elements) {
        d_data[idx] = make_cuFloatComplex(
            cuCrealf(d_data[idx]) * scale,
            cuCimagf(d_data[idx]) * scale
        );
    }
}

__global__ void copy_conv_result_to_batch_kernel(
    const cuFloatComplex* d_conv_result_batch,
    cuFloatComplex* d_g_batch,
    int conv_len,
    int conv_out_len,
    int frame_num
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = frame_num * conv_out_len;
    if (idx < total_elements) {
        int fi = idx / conv_out_len;
        int i = idx % conv_out_len;
        d_g_batch[idx] = d_conv_result_batch[fi * conv_len + i];
    }
}

__global__ void prepare_pn420_batch_kernel(
    const cuFloatComplex* __restrict__ d_pn420_bz,
    const int* __restrict__ d_frame,
    int frame_num,
    int frame_syn,
    cuFloatComplex* __restrict__ d_pn420_batch,
    int is_second_pass
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = frame_num * 420;
    if (idx < total_elements) {
        int fi = idx / 420;
        int i = idx % 420;

        int frame_n;
        if (is_second_pass == 0) {
            frame_n = d_frame[fi];
        } else {
            frame_n = (frame_syn + fi + 2) % 225;
            if (frame_n == 0) frame_n = 225;
        }

        int linear_idx = i * 225 + (frame_n - 1);
        d_pn420_batch[idx] = d_pn420_bz[linear_idx];
    }
}

__global__ void extract_data_jun_batch_kernel(
    const cuFloatComplex* __restrict__ d_dat,
    int frame_num,
    int delta,
    cuFloatComplex* __restrict__ d_data_jun) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 0 && idx < 3780 * frame_num) {
        int fi = idx / 3780;
        int inner_idx = idx % 3780;
        int data_idx = 4200 * (fi + 1) + delta + 420 + inner_idx;
        d_data_jun[idx] = d_dat[data_idx];
    }
}

__global__ void subtract_and_add_batch_kernel(
    const cuFloatComplex* __restrict__ d_dat,
    const cuFloatComplex* __restrict__ d_g1_batch,
    const cuFloatComplex* __restrict__ d_g2_batch,
    int frame_num,
    int delta,
    cuFloatComplex* __restrict__ d_data_jun) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 0 && idx < 254 * frame_num) {
        int fi = idx / 254;
        int inner_idx = idx % 254;

        int dat1_idx = 4200 * (fi + 1) + delta + 420 + inner_idx;
        int g1_idx = fi * 674 + 420 + inner_idx;
        cuFloatComplex x1 = make_cuFloatComplex(
            cuCrealf(d_dat[dat1_idx]) - cuCrealf(d_g1_batch[g1_idx]),
            cuCimagf(d_dat[dat1_idx]) - cuCimagf(d_g1_batch[g1_idx])
        );

        int dat2_idx = 4200 * (fi + 2) + delta + inner_idx;
        int g2_idx = fi * 674 + inner_idx;
        cuFloatComplex x2 = make_cuFloatComplex(
            cuCrealf(d_dat[dat2_idx]) - cuCrealf(d_g2_batch[g2_idx]),
            cuCimagf(d_dat[dat2_idx]) - cuCimagf(d_g2_batch[g2_idx])
        );

        d_data_jun[fi * 3780 + inner_idx] = make_cuFloatComplex(
            cuCrealf(x1) + cuCrealf(x2),
            cuCimagf(x1) + cuCimagf(x2)
        );
    }
}

__global__ void pad_hop_freq_batch_kernel(
    const cuFloatComplex* __restrict__ d_hop_freq_batch,
    int frame_num,
    cuFloatComplex* __restrict__ d_hop_freq_padded_batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 0 && idx < 3780 * frame_num) {
        int fi = idx / 3780;
        int inner_idx = idx % 3780;

        if (inner_idx < 255) {
            d_hop_freq_padded_batch[idx] = d_hop_freq_batch[fi * 255 + inner_idx];
        } else {
            d_hop_freq_padded_batch[idx] = make_cuFloatComplex(0.0f, 0.0f);
        }
    }
}

__global__ void copy_pn420_to_refclrdata_kernel(const cuFloatComplex* d_pn420_bz,
                                                    const int* d_frame,
                                                    int frame_num,
                                                    cuFloatComplex* d_refclrdata) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 0 && idx < 420 * frame_num) {
        int fi = idx / 420;
        int pn_idx = idx % 420;
        int frame_n = d_frame[fi];
        int col = frame_n - 1;
        int linear_idx = pn_idx * 225 + col;

        cuFloatComplex pn_val = d_pn420_bz[linear_idx];
        float scale = sqrtf(2.0f);

        d_refclrdata[fi * 4200 + pn_idx] = make_cuFloatComplex(
            scale * cuCrealf(pn_val),
            scale * cuCimagf(pn_val)
        );
    }
}

__global__ void copy_ifft_to_refclrdata_kernel(const cuFloatComplex* d_tmp,
                                                  int frame_num,
                                                  cuFloatComplex* d_refclrdata) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 0 && idx < 3780 * frame_num) {
        int fi = idx / 3780;
        int inner_idx = idx % 3780;

        d_refclrdata[fi * 4200 + 420 + inner_idx] = d_tmp[idx];
    }
}

__global__ void extract_dat255_batch_kernel(
    const cuFloatComplex* __restrict__ d_dat,
    int frame_num,
    int delta,
    cuFloatComplex* __restrict__ d_dat255_batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 0 && idx < 255 * frame_num) {
        int fi = idx / 255;
        int i = idx % 255;
        int data_idx = 4200 * (fi + 1) + delta + 165 + i;
        d_dat255_batch[idx] = d_dat[data_idx];
    }
}

__global__ void extract_pn255_batch_kernel(
    const cuFloatComplex* __restrict__ d_pn420_bz,
    const int* __restrict__ d_frame,
    int frame_num,
    cuFloatComplex* __restrict__ d_pn255_batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 0 && idx < 255 * frame_num) {
        int fi = idx / 255;
        int i = idx % 255;
        int frame_n = d_frame[fi];
        int linear_idx = (165 + i) * 225 + (frame_n - 1);
        d_pn255_batch[idx] = d_pn420_bz[linear_idx];
    }
}

RefClearGPUCache* create_ref_clear_gpu_cache(int max_frame_num, HEstimateImprovedGPUCache* external_h_est_improved_cache) {
    RefClearGPUCache* cache = (RefClearGPUCache*)malloc(sizeof(RefClearGPUCache));
    if (!cache) {
        return NULL;
    }

    cache->max_frame_num = max_frame_num;
    cache->is_valid = false;

    size_t dat_size = (max_frame_num + 2) * 4200 * sizeof(cuFloatComplex);
    size_t pn420_size = 420 * 225 * sizeof(cuFloatComplex);
    size_t refclrdata_size = 4200 * max_frame_num * sizeof(cuFloatComplex);

    const int conv_fft_len = 1024;
    size_t conv_fft_size = conv_fft_len * sizeof(cuFloatComplex);

    CUDA_CHECK(cudaMalloc(&cache->d_dat, dat_size));
    CUDA_CHECK(cudaMalloc(&cache->d_pn420_bz, pn420_size));
    CUDA_CHECK(cudaMalloc(&cache->d_refclrdata, refclrdata_size));

    CUDA_CHECK(cudaMalloc(&cache->d_h_fft_freq, 3780 * max_frame_num * sizeof(cuFloatComplex)));

    size_t conv_batch_size = max_frame_num * conv_fft_size * sizeof(cuFloatComplex);
    CUDA_CHECK(cudaMalloc(&cache->d_conv_pn420_padded_batch, conv_batch_size));
    CUDA_CHECK(cudaMalloc(&cache->d_conv_hf_padded_batch, conv_batch_size));
    CUDA_CHECK(cudaMalloc(&cache->d_conv_pn420_fft_batch, conv_batch_size));
    CUDA_CHECK(cudaMalloc(&cache->d_conv_hf_fft_batch, conv_batch_size));
    CUDA_CHECK(cudaMalloc(&cache->d_conv_result_batch, conv_batch_size));

    size_t conv_out_len = 420 + 255 - 1;
    CUDA_CHECK(cudaMalloc(&cache->d_g1_batch, conv_out_len * max_frame_num * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&cache->d_g2_batch, conv_out_len * max_frame_num * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&cache->d_hop_freq_batch, 255 * max_frame_num * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&cache->d_hop_freq_padded_batch, 3780 * max_frame_num * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&cache->d_data_jun_batch, 3780 * max_frame_num * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&cache->d_data_in_fft_freq_batch, 3780 * max_frame_num * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&cache->d_tmp_batch, 3780 * max_frame_num * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&cache->d_frame, max_frame_num * sizeof(int)));

    int rank = 1;
    int fft_size = 3780;
    int inembed[] = { fft_size };
    int onembed[] = { fft_size };
    int stride = 1;
    int dist = fft_size;
    int batch_size = max_frame_num;

    CUFFT_CHECK(cufftPlanMany(&cache->fft_plan_3780_batch, rank, &fft_size,
        inembed, stride, dist,
        onembed, stride, dist,
        CUFFT_C2C, batch_size));
    CUFFT_CHECK(cufftPlanMany(&cache->ifft_plan_3780_batch, rank, &fft_size,
        inembed, stride, dist,
        onembed, stride, dist,
        CUFFT_C2C, batch_size));

    int conv_rank = 1;
    int conv_fft_size_val = conv_fft_len;
    int conv_inembed[] = { conv_fft_size_val };
    int conv_onembed[] = { conv_fft_size_val };
    int conv_stride = 1;
    int conv_dist = conv_fft_size_val;

    CUFFT_CHECK(cufftPlanMany(&cache->fft_plan_conv_batch, conv_rank, &conv_fft_size_val,
        conv_inembed, conv_stride, conv_dist,
        conv_onembed, conv_stride, conv_dist,
        CUFFT_C2C, batch_size));
    CUFFT_CHECK(cufftPlanMany(&cache->ifft_plan_conv_batch, conv_rank, &conv_fft_size_val,
        conv_inembed, conv_stride, conv_dist,
        conv_onembed, conv_stride, conv_dist,
        CUFFT_C2C, batch_size));

    cache->h_est_improved_cache = external_h_est_improved_cache ? external_h_est_improved_cache : create_h_estimate_improved_gpu_cache_with_batch(max_frame_num);

    cache->pn420_initialized = false;
    cache->is_valid = true;
    return cache;
}

void destroy_ref_clear_gpu_cache(RefClearGPUCache* cache, bool free_h_est_improved_cache) {
    if (!cache) {
        return;
    }
    if (cache->is_valid) {
        CUDA_CHECK(cudaFree(cache->d_dat));
        CUDA_CHECK(cudaFree(cache->d_pn420_bz));
        CUDA_CHECK(cudaFree(cache->d_refclrdata));

        CUDA_CHECK(cudaFree(cache->d_h_fft_freq));

        CUDA_CHECK(cudaFree(cache->d_conv_pn420_padded_batch));
        CUDA_CHECK(cudaFree(cache->d_conv_hf_padded_batch));
        CUDA_CHECK(cudaFree(cache->d_conv_pn420_fft_batch));
        CUDA_CHECK(cudaFree(cache->d_conv_hf_fft_batch));
        CUDA_CHECK(cudaFree(cache->d_conv_result_batch));

        CUDA_CHECK(cudaFree(cache->d_g1_batch));
        CUDA_CHECK(cudaFree(cache->d_g2_batch));
        CUDA_CHECK(cudaFree(cache->d_hop_freq_batch));
        CUDA_CHECK(cudaFree(cache->d_hop_freq_padded_batch));
        CUDA_CHECK(cudaFree(cache->d_data_jun_batch));
        CUDA_CHECK(cudaFree(cache->d_data_in_fft_freq_batch));
        CUDA_CHECK(cudaFree(cache->d_tmp_batch));
        CUDA_CHECK(cudaFree(cache->d_frame));

        CUFFT_CHECK(cufftDestroy(cache->fft_plan_3780_batch));
        CUFFT_CHECK(cufftDestroy(cache->ifft_plan_3780_batch));
        CUFFT_CHECK(cufftDestroy(cache->fft_plan_conv_batch));
        CUFFT_CHECK(cufftDestroy(cache->ifft_plan_conv_batch));

        if (free_h_est_improved_cache && cache->h_est_improved_cache) {
            destroy_h_estimate_improved_gpu_cache(cache->h_est_improved_cache);
        }
    }
    free(cache);
}

std::vector<std::vector<std::complex<float>>> get_pn420_bz() {
    std::vector<std::vector<std::complex<float>>> pn420_bz(420, std::vector<std::complex<float>>(225));

    float scale = (1.0f + 1.0f) / sqrtf(2.0f);

    for (int i = 0; i < 420; i++) {
        for (int j = 0; j < 225; j++) {
            float val = (pn420_01[i][j] == 0) ? 1.0f : -1.0f;
            pn420_bz[i][j] = std::complex<float>(val * scale / 2.0f, val * scale / 2.0f);
        }
    }

    return pn420_bz;
}

RefClearResult ref_clear_gpu_batch(const std::vector<std::complex<float>>& dat,
                                     int frame_syn,
                                     int frame_num,
                                     const std::vector<std::vector<std::complex<float>>>& pn420_bz,
                                     RefClearGPUCache* cache) {
    RefClearResult result;

    if (!cache || !cache->is_valid) {
        return result;
    }

    int delta = 0;

    std::vector<int> frame(frame_num);
    for (int fi = 0; fi < frame_num; fi++) {
        int frame_n = (frame_syn + fi + 1) % 225;
        if (frame_n == 0) frame_n = 225;
        frame[fi] = frame_n;
    }
    CUDA_CHECK(cudaMemcpy(cache->d_frame, frame.data(), frame_num * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize3780 = (3780 * frame_num + blockSize - 1) / blockSize;
    int gridSize254 = (254 * frame_num + blockSize - 1) / blockSize;
    float scale_fft = sqrtf(1.0);
    float scale_ifft = 1.0f / 3780.0f;
    float scale_conv = 1.0f / 1024.0f;
    const int conv_out_len = 420 + 255 - 1;

    size_t dat_size = dat.size() * sizeof(cuFloatComplex);
    std::vector<cuFloatComplex> h_dat_cpu(dat.size());
    for (size_t i = 0; i < dat.size(); i++) {
        h_dat_cpu[i] = make_cuFloatComplex(dat[i].real(), dat[i].imag());
    }
    CUDA_CHECK(cudaMemcpy(cache->d_dat, h_dat_cpu.data(), dat_size, cudaMemcpyHostToDevice));

    if (!cache->pn420_initialized) {
        size_t pn420_size = 420 * 225 * sizeof(cuFloatComplex);
        std::vector<cuFloatComplex> h_pn420_bz_cpu(420 * 225);
        for (int i = 0; i < 420; i++) {
            for (int j = 0; j < 225; j++) {
                h_pn420_bz_cpu[i * 225 + j] = make_cuFloatComplex(
                    pn420_bz[i][j].real(), pn420_bz[i][j].imag()
                );
            }
        }
        CUDA_CHECK(cudaMemcpy(cache->d_pn420_bz, h_pn420_bz_cpu.data(), pn420_size, cudaMemcpyHostToDevice));
        cache->pn420_initialized = true;
    }

    std::vector<cuFloatComplex> h_hop_freq_batch(255 * frame_num);
    std::vector<cuFloatComplex> dat255_batch(255 * frame_num);
    std::vector<cuFloatComplex> pn255_batch(255 * frame_num);

    for (int fi = 0; fi < frame_num; fi++) {
        int frame_n = frame[fi];
        for (int i = 0; i < 255; i++) {
            pn255_batch[fi * 255 + i] = make_cuFloatComplex(
                pn420_bz[165 + i][frame_n - 1].real(),
                pn420_bz[165 + i][frame_n - 1].imag()
            );
            int data_idx = 4200 * (fi + 1) + delta + 165 + i;
            dat255_batch[fi * 255 + i] = make_cuFloatComplex(
                dat[data_idx].real(),
                dat[data_idx].imag()
            );
        }
    }

    h_estimate_improved_gpu_batch_with_cache(dat255_batch, pn255_batch, frame_num, h_hop_freq_batch, cache->h_est_improved_cache);
    CUDA_CHECK(cudaMemcpy(cache->d_hop_freq_batch, h_hop_freq_batch.data(), 255 * frame_num * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    cuFloatComplex* d_pn420_batch = NULL;
    CUDA_CHECK(cudaMalloc(&d_pn420_batch, frame_num * 420 * sizeof(cuFloatComplex)));

    int gridSizePn = (frame_num * 420 + blockSize - 1) / blockSize;
    prepare_pn420_batch_kernel<<<gridSizePn, blockSize>>>(
        cache->d_pn420_bz,
        cache->d_frame,
        frame_num,
        frame_syn,
        d_pn420_batch,
        0
    );
    CUDA_CHECK(cudaGetLastError());

    int gridSizeConvBatch = (frame_num * 1024 + blockSize - 1) / blockSize;
    pad_arrays_for_conv_batch_kernel<<<gridSizeConvBatch, blockSize>>>(
        d_pn420_batch,
        cache->d_hop_freq_batch,
        cache->d_conv_pn420_padded_batch,
        cache->d_conv_hf_padded_batch,
        420, 255, 1024,
        frame_num
    );
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(cache->fft_plan_conv_batch, (cufftComplex*)cache->d_conv_pn420_padded_batch, (cufftComplex*)cache->d_conv_pn420_fft_batch, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(cache->fft_plan_conv_batch, (cufftComplex*)cache->d_conv_hf_padded_batch, (cufftComplex*)cache->d_conv_hf_fft_batch, CUFFT_FORWARD));

    complex_multiply_batch_kernel<<<gridSizeConvBatch, blockSize>>>(
        cache->d_conv_pn420_fft_batch,
        cache->d_conv_hf_fft_batch,
        cache->d_conv_result_batch,
        1024,
        frame_num
    );
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(cache->ifft_plan_conv_batch, (cufftComplex*)cache->d_conv_result_batch, (cufftComplex*)cache->d_conv_result_batch, CUFFT_INVERSE));

    normalize_ifft_conv_batch_kernel<<<gridSizeConvBatch, blockSize>>>(
        cache->d_conv_result_batch,
        1024,
        frame_num,
        scale_conv
    );
    CUDA_CHECK(cudaGetLastError());

    copy_conv_result_to_batch_kernel<<<(frame_num * conv_out_len + blockSize - 1) / blockSize, blockSize>>>(
        cache->d_conv_result_batch,
        cache->d_g1_batch,
        1024,
        conv_out_len,
        frame_num
    );
    CUDA_CHECK(cudaGetLastError());

    prepare_pn420_batch_kernel<<<gridSizePn, blockSize>>>(
        cache->d_pn420_bz,
        cache->d_frame,
        frame_num,
        frame_syn,
        d_pn420_batch,
        1
    );
    CUDA_CHECK(cudaGetLastError());

    pad_arrays_for_conv_batch_kernel<<<gridSizeConvBatch, blockSize>>>(
        d_pn420_batch,
        cache->d_hop_freq_batch,
        cache->d_conv_pn420_padded_batch,
        cache->d_conv_hf_padded_batch,
        420, 255, 1024,
        frame_num
    );
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(cache->fft_plan_conv_batch, (cufftComplex*)cache->d_conv_pn420_padded_batch, (cufftComplex*)cache->d_conv_pn420_fft_batch, CUFFT_FORWARD));

    complex_multiply_batch_kernel<<<gridSizeConvBatch, blockSize>>>(
        cache->d_conv_pn420_fft_batch,
        cache->d_conv_hf_fft_batch,
        cache->d_conv_result_batch,
        1024,
        frame_num
    );
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(cache->ifft_plan_conv_batch, (cufftComplex*)cache->d_conv_result_batch, (cufftComplex*)cache->d_conv_result_batch, CUFFT_INVERSE));

    normalize_ifft_conv_batch_kernel<<<gridSizeConvBatch, blockSize>>>(
        cache->d_conv_result_batch,
        1024,
        frame_num,
        scale_conv
    );
    CUDA_CHECK(cudaGetLastError());

    copy_conv_result_to_batch_kernel<<<(frame_num * conv_out_len + blockSize - 1) / blockSize, blockSize>>>(
        cache->d_conv_result_batch,
        cache->d_g2_batch,
        1024,
        conv_out_len,
        frame_num
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_pn420_batch));

    extract_data_jun_batch_kernel<<<gridSize3780, blockSize>>>(
        cache->d_dat, frame_num, delta, cache->d_data_jun_batch);
    CUDA_CHECK(cudaGetLastError());

    subtract_and_add_batch_kernel<<<gridSize254, blockSize>>>(
        cache->d_dat, cache->d_g1_batch, cache->d_g2_batch,
        frame_num, delta, cache->d_data_jun_batch);
    CUDA_CHECK(cudaGetLastError());

    pad_hop_freq_batch_kernel<<<gridSize3780, blockSize>>>(
        cache->d_hop_freq_batch, frame_num, cache->d_hop_freq_padded_batch);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(cache->fft_plan_3780_batch,
        (cufftComplex*)cache->d_data_jun_batch,
        (cufftComplex*)cache->d_data_in_fft_freq_batch,
        CUFFT_FORWARD));

    CUFFT_CHECK(cufftExecC2C(cache->fft_plan_3780_batch,
        (cufftComplex*)cache->d_hop_freq_padded_batch,
        (cufftComplex*)cache->d_h_fft_freq,
        CUFFT_FORWARD));

    int gridSizeBatch = (frame_num * 3780 + blockSize - 1) / blockSize;
    complex_divide_batch_kernel<<<gridSizeBatch, blockSize>>>(
        cache->d_data_in_fft_freq_batch,
        cache->d_h_fft_freq,
        cache->d_data_in_fft_freq_batch,
        frame_num,
        3780,
        scale_fft);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(cache->ifft_plan_3780_batch,
        (cufftComplex*)cache->d_data_in_fft_freq_batch,
        (cufftComplex*)cache->d_tmp_batch,
        CUFFT_INVERSE));

    normalize_ifft_batch_kernel<<<gridSizeBatch, blockSize>>>(
        cache->d_tmp_batch, frame_num, 3780, scale_ifft);
    CUDA_CHECK(cudaGetLastError());

    copy_pn420_to_refclrdata_kernel<<<(420 * frame_num + blockSize - 1) / blockSize, blockSize>>>(
        cache->d_pn420_bz, cache->d_frame, frame_num, cache->d_refclrdata);
    CUDA_CHECK(cudaGetLastError());

    copy_ifft_to_refclrdata_kernel<<<(3780 * frame_num + blockSize - 1) / blockSize, blockSize>>>(
        cache->d_tmp_batch, frame_num, cache->d_refclrdata);
    CUDA_CHECK(cudaGetLastError());

    std::vector<std::complex<float>> refclrdata(4200 * frame_num);
    std::vector<cuFloatComplex> h_refclrdata_cpu(4200 * frame_num);
    CUDA_CHECK(cudaMemcpy(h_refclrdata_cpu.data(), cache->d_refclrdata, 4200 * frame_num * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 4200 * frame_num; i++) {
        refclrdata[i] = std::complex<float>(
            cuCrealf(h_refclrdata_cpu[i]), cuCimagf(h_refclrdata_cpu[i])
        );
    }

    std::vector<std::complex<float>> data_in_fft_freq(3780 * frame_num);
    std::vector<cuFloatComplex> h_data_in_fft_freq_cpu(3780 * frame_num);
    CUDA_CHECK(cudaMemcpy(h_data_in_fft_freq_cpu.data(), cache->d_data_in_fft_freq_batch, 3780 * frame_num * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 3780 * frame_num; i++) {
        data_in_fft_freq[i] = std::complex<float>(
            cuCrealf(h_data_in_fft_freq_cpu[i]), cuCimagf(h_data_in_fft_freq_cpu[i])
        );
    }

    result.refclrdata = refclrdata;
    result.data_in_fft_freq = data_in_fft_freq;
    result.delta = 0;

    return result;
}

void export_constellation_data_binary(const std::vector<std::complex<float>>& data_in_fft_freq,
                                        const std::string& filename_freq) {
    std::ofstream file_freq(filename_freq, std::ios::binary);
    if (file_freq.is_open()) {
        size_t num_elements = data_in_fft_freq.size();
        file_freq.write(reinterpret_cast<const char*>(data_in_fft_freq.data()),
                        num_elements * sizeof(std::complex<float>));
        file_freq.close();
        std::cout << "Exported frequency domain constellation data (binary) to " << filename_freq << std::endl;
    }
}

void generate_constellation_plot_script_binary(const std::string& filename_freq,
                                                 const std::string& script_filename) {
    std::ofstream script(script_filename);
    if (script.is_open()) {
        script << "import numpy as np\n";
        script << "import matplotlib.pyplot as plt\n\n";

        script << "def load_binary_data(filename):\n";
        script << "    data = np.fromfile(filename, dtype=np.complex64)\n";
        script << "    return data\n\n";

        script << "def plot_constellation_binary(filename, title_suffix):\n";
        script << "    data = load_binary_data(filename)\n";
        script << "    real = data.real\n";
        script << "    imag = data.imag\n\n";

        script << "    fig, axes = plt.subplots(1, 2, figsize=(20, 10))\n\n";

        script << "    axes[0].scatter(real, imag, s=1, alpha=0.1)\n";
        script << "    axes[0].axis('equal')\n";
        script << "    axes[0].grid(True)\n";
        script << "    axes[0].set_xlabel('Real')\n";
        script << "    axes[0].set_ylabel('Imaginary')\n";
        script << "    axes[0].set_title(f'Constellation Diagram - {title_suffix}')\n\n";

        script << "    subcarrier_start = 890\n";
        script << "    subcarrier_end = 1910\n";
        script << "    frame_size = 3780\n";
        script << "    num_frames = len(data) // frame_size\n\n";

        script << "    sub_data = []\n";
        script << "    for i in range(num_frames):\n";
        script << "        start = i * frame_size + subcarrier_start\n";
        script << "        end = i * frame_size + subcarrier_end\n";
        script << "        sub_data.append(data[start:end])\n";
        script << "    sub_data = np.concatenate(sub_data)\n\n";

        script << "    axes[1].scatter(sub_data.real, sub_data.imag, s=1, alpha=0.1)\n";
        script << "    axes[1].axis('equal')\n";
        script << "    axes[1].grid(True)\n";
        script << "    axes[1].set_xlabel('Real')\n";
        script << "    axes[1].set_ylabel('Imaginary')\n";
        script << "    axes[1].set_title(f'Constellation Diagram (Subcarriers {subcarrier_start}-{subcarrier_end}) - {title_suffix}')\n\n";

        script << "    plt.tight_layout()\n";
        script << "    plt.show()\n\n";

        script << "if __name__ == '__main__':\n";
        script << "    plot_constellation_binary('" << filename_freq << "', 'Frequency Domain Channel Estimation')\n";

        script.close();
        std::cout << "Generated constellation plot script: " << script_filename << std::endl;
    }
}

RefClearResultDevice ref_clear_gpu_batch_device(
                                    const cuFloatComplex* d_dat_src,
                                    int dat_offset,
                                    int dat_total_size,
                                    int frame_syn,
                                    int frame_num,
                                    const std::vector<std::vector<std::complex<float>>>& pn420_bz,
                                    RefClearGPUCache* cache) {
    RefClearResultDevice result;
    result.d_refclrdata = nullptr;
    result.d_data_in_fft_freq = nullptr;
    result.frame_num = frame_num;
    result.delta = 0;

    if (!cache || !cache->is_valid) {
        return result;
    }

    int delta = 0;

    std::vector<int> frame(frame_num);
    for (int fi = 0; fi < frame_num; fi++) {
        int frame_n = (frame_syn + fi + 1) % 225;
        if (frame_n == 0) frame_n = 225;
        frame[fi] = frame_n;
    }
    CUDA_CHECK(cudaMemcpy(cache->d_frame, frame.data(), frame_num * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize3780 = (3780 * frame_num + blockSize - 1) / blockSize;
    int gridSize254 = (254 * frame_num + blockSize - 1) / blockSize;
    float scale_fft = sqrtf(1.0);
    float scale_ifft = 1.0f / 3780.0f;
    float scale_conv = 1.0f / 1024.0f;
    const int conv_out_len = 420 + 255 - 1;

    size_t dat_size = dat_total_size * sizeof(cuFloatComplex);
    CUDA_CHECK(cudaMemcpy(cache->d_dat, d_dat_src + dat_offset, dat_size, cudaMemcpyDeviceToDevice));

    if (!cache->pn420_initialized) {
        size_t pn420_size = 420 * 225 * sizeof(cuFloatComplex);
        std::vector<cuFloatComplex> h_pn420_bz_cpu(420 * 225);
        for (int i = 0; i < 420; i++) {
            for (int j = 0; j < 225; j++) {
                h_pn420_bz_cpu[i * 225 + j] = make_cuFloatComplex(
                    pn420_bz[i][j].real(), pn420_bz[i][j].imag()
                );
            }
        }
        CUDA_CHECK(cudaMemcpy(cache->d_pn420_bz, h_pn420_bz_cpu.data(), pn420_size, cudaMemcpyHostToDevice));
        cache->pn420_initialized = true;
    }

    HEstimateImprovedGPUCache* h_est_cache = cache->h_est_improved_cache;

    int gridSize255 = (255 * frame_num + blockSize - 1) / blockSize;
    extract_dat255_batch_kernel<<<gridSize255, blockSize>>>(
        cache->d_dat, frame_num, delta, h_est_cache->d_dat255_batch);
    CUDA_CHECK(cudaGetLastError());

    extract_pn255_batch_kernel<<<gridSize255, blockSize>>>(
        cache->d_pn420_bz, cache->d_frame, frame_num, h_est_cache->d_pn255_batch);
    CUDA_CHECK(cudaGetLastError());

    std::vector<cuFloatComplex> h_hop_freq_batch;
    h_estimate_improved_gpu_batch_device_with_cache(frame_num, h_hop_freq_batch, h_est_cache);

    CUDA_CHECK(cudaMemcpy(cache->d_hop_freq_batch, h_hop_freq_batch.data(), 255 * frame_num * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    cuFloatComplex* d_pn420_batch = NULL;
    CUDA_CHECK(cudaMalloc(&d_pn420_batch, frame_num * 420 * sizeof(cuFloatComplex)));

    int gridSizePn = (frame_num * 420 + blockSize - 1) / blockSize;
    prepare_pn420_batch_kernel<<<gridSizePn, blockSize>>>(
        cache->d_pn420_bz,
        cache->d_frame,
        frame_num,
        frame_syn,
        d_pn420_batch,
        0
    );
    CUDA_CHECK(cudaGetLastError());

    int gridSizeConvBatch = (frame_num * 1024 + blockSize - 1) / blockSize;
    pad_arrays_for_conv_batch_kernel<<<gridSizeConvBatch, blockSize>>>(
        d_pn420_batch,
        cache->d_hop_freq_batch,
        cache->d_conv_pn420_padded_batch,
        cache->d_conv_hf_padded_batch,
        420, 255, 1024,
        frame_num
    );
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(cache->fft_plan_conv_batch, (cufftComplex*)cache->d_conv_pn420_padded_batch, (cufftComplex*)cache->d_conv_pn420_fft_batch, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(cache->fft_plan_conv_batch, (cufftComplex*)cache->d_conv_hf_padded_batch, (cufftComplex*)cache->d_conv_hf_fft_batch, CUFFT_FORWARD));

    complex_multiply_batch_kernel<<<gridSizeConvBatch, blockSize>>>(
        cache->d_conv_pn420_fft_batch,
        cache->d_conv_hf_fft_batch,
        cache->d_conv_result_batch,
        1024,
        frame_num
    );
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(cache->ifft_plan_conv_batch, (cufftComplex*)cache->d_conv_result_batch, (cufftComplex*)cache->d_conv_result_batch, CUFFT_INVERSE));

    normalize_ifft_conv_batch_kernel<<<gridSizeConvBatch, blockSize>>>(
        cache->d_conv_result_batch,
        1024,
        frame_num,
        scale_conv
    );
    CUDA_CHECK(cudaGetLastError());

    copy_conv_result_to_batch_kernel<<<(frame_num * conv_out_len + blockSize - 1) / blockSize, blockSize>>>(
        cache->d_conv_result_batch,
        cache->d_g1_batch,
        1024,
        conv_out_len,
        frame_num
    );
    CUDA_CHECK(cudaGetLastError());

    prepare_pn420_batch_kernel<<<gridSizePn, blockSize>>>(
        cache->d_pn420_bz,
        cache->d_frame,
        frame_num,
        frame_syn,
        d_pn420_batch,
        1
    );
    CUDA_CHECK(cudaGetLastError());

    pad_arrays_for_conv_batch_kernel<<<gridSizeConvBatch, blockSize>>>(
        d_pn420_batch,
        cache->d_hop_freq_batch,
        cache->d_conv_pn420_padded_batch,
        cache->d_conv_hf_padded_batch,
        420, 255, 1024,
        frame_num
    );
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(cache->fft_plan_conv_batch, (cufftComplex*)cache->d_conv_pn420_padded_batch, (cufftComplex*)cache->d_conv_pn420_fft_batch, CUFFT_FORWARD));

    complex_multiply_batch_kernel<<<gridSizeConvBatch, blockSize>>>(
        cache->d_conv_pn420_fft_batch,
        cache->d_conv_hf_fft_batch,
        cache->d_conv_result_batch,
        1024,
        frame_num
    );
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(cache->ifft_plan_conv_batch, (cufftComplex*)cache->d_conv_result_batch, (cufftComplex*)cache->d_conv_result_batch, CUFFT_INVERSE));

    normalize_ifft_conv_batch_kernel<<<gridSizeConvBatch, blockSize>>>(
        cache->d_conv_result_batch,
        1024,
        frame_num,
        scale_conv
    );
    CUDA_CHECK(cudaGetLastError());

    copy_conv_result_to_batch_kernel<<<(frame_num * conv_out_len + blockSize - 1) / blockSize, blockSize>>>(
        cache->d_conv_result_batch,
        cache->d_g2_batch,
        1024,
        conv_out_len,
        frame_num
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_pn420_batch));

    extract_data_jun_batch_kernel<<<gridSize3780, blockSize>>>(
        cache->d_dat, frame_num, delta, cache->d_data_jun_batch);
    CUDA_CHECK(cudaGetLastError());

    subtract_and_add_batch_kernel<<<gridSize254, blockSize>>>(
        cache->d_dat, cache->d_g1_batch, cache->d_g2_batch,
        frame_num, delta, cache->d_data_jun_batch);
    CUDA_CHECK(cudaGetLastError());

    pad_hop_freq_batch_kernel<<<gridSize3780, blockSize>>>(
        cache->d_hop_freq_batch, frame_num, cache->d_hop_freq_padded_batch);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(cache->fft_plan_3780_batch,
        (cufftComplex*)cache->d_data_jun_batch,
        (cufftComplex*)cache->d_data_in_fft_freq_batch,
        CUFFT_FORWARD));

    CUFFT_CHECK(cufftExecC2C(cache->fft_plan_3780_batch,
        (cufftComplex*)cache->d_hop_freq_padded_batch,
        (cufftComplex*)cache->d_h_fft_freq,
        CUFFT_FORWARD));

    int gridSizeBatch = (frame_num * 3780 + blockSize - 1) / blockSize;
    complex_divide_batch_kernel<<<gridSizeBatch, blockSize>>>(
        cache->d_data_in_fft_freq_batch,
        cache->d_h_fft_freq,
        cache->d_data_in_fft_freq_batch,
        frame_num,
        3780,
        scale_fft);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(cache->ifft_plan_3780_batch,
        (cufftComplex*)cache->d_data_in_fft_freq_batch,
        (cufftComplex*)cache->d_tmp_batch,
        CUFFT_INVERSE));

    normalize_ifft_batch_kernel<<<gridSizeBatch, blockSize>>>(
        cache->d_tmp_batch, frame_num, 3780, scale_ifft);
    CUDA_CHECK(cudaGetLastError());

    copy_pn420_to_refclrdata_kernel<<<(420 * frame_num + blockSize - 1) / blockSize, blockSize>>>(
        cache->d_pn420_bz, cache->d_frame, frame_num, cache->d_refclrdata);
    CUDA_CHECK(cudaGetLastError());

    copy_ifft_to_refclrdata_kernel<<<(3780 * frame_num + blockSize - 1) / blockSize, blockSize>>>(
        cache->d_tmp_batch, frame_num, cache->d_refclrdata);
    CUDA_CHECK(cudaGetLastError());

    result.d_refclrdata = cache->d_refclrdata;
    result.d_data_in_fft_freq = cache->d_data_in_fft_freq_batch;
    result.frame_num = frame_num;
    result.delta = 0;

    return result;
}
