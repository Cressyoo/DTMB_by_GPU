#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <sys/stat.h>
#include <sys/types.h>

#include <cuda_runtime.h>
#include <cufft.h>
#include <nvtx3/nvToolsExt.h>
#include "read_dtmb_signal.h"
#include "dtmb_frame_sync.h"
#include "dtmb_freq_estimate.h"
#include "ref_clear.h"
#include "shared_memory.h"
#include "cuda_downsampling.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#define MKDIR(path) mkdir(path, 0755)
#endif

struct StageTimers {
    double data_transfer;
    double downsample;
    double frame_sync;
    double freq_estimate;
    double freq_compensate;
    double ref_clear;
    double export_d2h;
    double other;

    StageTimers() : data_transfer(0), downsample(0), frame_sync(0), freq_estimate(0),
                    freq_compensate(0), ref_clear(0), export_d2h(0), other(0) {}
};

struct ExportTask {
    std::vector<std::complex<float>> data_in_fft_freq;
    std::vector<std::complex<float>> ref_extracted;
    std::vector<std::complex<float>> monitor_ch1;
    std::vector<std::complex<float>> monitor_ch2;
    std::string constellation_filename;
    std::string dtmb_result_filename;
};

static std::queue<ExportTask> g_export_queue;
static std::mutex g_export_mutex;
static std::condition_variable g_export_cv;
static std::atomic<bool> g_export_done(false);

void export_thread_func() {
    while (true) {
        ExportTask task;
        {
            std::unique_lock<std::mutex> lock(g_export_mutex);
            g_export_cv.wait(lock, [] { return !g_export_queue.empty() || g_export_done.load(); });
            if (g_export_queue.empty() && g_export_done.load()) break;
            task = std::move(g_export_queue.front());
            g_export_queue.pop();
        }

        if (!task.constellation_filename.empty() && !task.data_in_fft_freq.empty()) {
            std::ofstream file_freq(task.constellation_filename, std::ios::binary);
            if (file_freq.is_open()) {
                file_freq.write(reinterpret_cast<const char*>(task.data_in_fft_freq.data()),
                              task.data_in_fft_freq.size() * sizeof(std::complex<float>));
                file_freq.close();
            }
        }

        if (!task.dtmb_result_filename.empty() && !task.ref_extracted.empty()) {
            std::ofstream dtmb_file(task.dtmb_result_filename, std::ios::binary);
            if (dtmb_file.is_open()) {
                dtmb_file.write(reinterpret_cast<const char*>(task.ref_extracted.data()),
                              task.ref_extracted.size() * sizeof(std::complex<float>));
                dtmb_file.write(reinterpret_cast<const char*>(task.monitor_ch1.data()),
                              task.monitor_ch1.size() * sizeof(std::complex<float>));
                dtmb_file.write(reinterpret_cast<const char*>(task.monitor_ch2.data()),
                              task.monitor_ch2.size() * sizeof(std::complex<float>));
                dtmb_file.close();
            }
        }
    }
}

int main(int argc, char* argv[]) {
    float originalSampleRate = SAMPLE_RATE;
    bool needsDownsampling = false;
    uint32_t dataMode = 0;

    std::cout << "========================================" << std::endl;
    std::cout << "  DTMB Real-Time Receiver (3-Channel)" << std::endl;
    std::cout << "  Optimized Pipeline Mode" << std::endl;
    std::cout << "  - Early unlock_rx() for sender overlap" << std::endl;
    std::cout << "  - GPU-only ref_clear (no D2H round-trip)" << std::endl;
    std::cout << "  - Async file export pipeline" << std::endl;
    std::cout << "  Mode will be auto-detected from sender" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;

    try {
        DTMBSignalParm signalParm = get_dtmb_signal_param();
        std::cout << "=== DTMB Signal Parameters ===" << std::endl;
        std::cout << "Frame header length: " << signalParm.frame_header_len << std::endl;
        std::cout << "Frame body length: " << signalParm.frame_body_len << std::endl;
        std::cout << "Frame length: " << signalParm.frame_len << std::endl;
        std::cout << "FFT size: " << signalParm.fft_size << std::endl;
        std::cout << "Target sample rate: " << signalParm.sample_rate / 1e6 << " MHz" << std::endl;
        std::cout << "Batch size: " << BATCH_SIZE << " samples (" << BATCH_SIZE / 4200 << " frames)" << std::endl << std::endl;

        std::vector<std::complex<float>> pn420;
        if (!read_pn420("pn420_0.mat", pn420)) {
            std::cerr << "Error: Failed to read PN420 sequence" << std::endl;
            return -1;
        }
        std::cout << "PN420 loaded: " << pn420.size() << " samples" << std::endl;

        int outputChannels = 3;
        HEstimateImprovedGPUCache* hEstImprovedCache = create_h_estimate_improved_gpu_cache_with_batch(256);
        DTMBGPUCache* dtmbCache = create_dtmb_gpu_cache(BATCH_SIZE, 3 * 4200);
        FreqEstimateGPUCache* freqCache = create_freq_estimate_gpu_cache(256);
        RefClearGPUCache* refClearCache = create_ref_clear_gpu_cache(256, hEstImprovedCache);
        FreqCompensateGPUCache* freqCompCache = create_freq_compensate_gpu_cache(outputChannels * BATCH_SIZE);

        FFTPlanCache* downsampleCache = nullptr;
        int inputBatchSize = BATCH_SIZE;

        if (!dtmbCache || !freqCache || !freqCompCache || !hEstImprovedCache || !refClearCache) {
            std::cerr << "Error: Failed to create GPU caches" << std::endl;
            if (dtmbCache) destroy_dtmb_gpu_cache(dtmbCache);
            if (freqCache) destroy_freq_estimate_gpu_cache(freqCache);
            if (refClearCache) destroy_ref_clear_gpu_cache(refClearCache, true);
            if (freqCompCache) destroy_freq_compensate_gpu_cache(freqCompCache);
            return -1;
        }

        std::cout << "Initializing PN420 BZ complex data..." << std::endl;
        init_pn420_bz_complex();

        std::cout << "Getting PN420 BZ data for ref_clear..." << std::endl;
        std::vector<std::vector<std::complex<float>>> pn420_bz = get_pn420_bz();
        std::cout << "PN420 BZ size: " << pn420_bz.size() << " x " << pn420_bz[0].size() << std::endl;

        cufftComplex* d_processedData = nullptr;
        CUDA_CHECK(cudaMalloc(&d_processedData, outputChannels * BATCH_SIZE * sizeof(cufftComplex)));

        cuFloatComplex* h_pinnedData = nullptr;
        cudaStream_t stream1;
        CUDA_CHECK(cudaStreamCreate(&stream1));

        std::thread export_thread(export_thread_func);

        SharedMemory shm;
        std::cout << std::endl << "Waiting for sender connection..." << std::endl;
        std::cout << "Please start the sender GUI..." << std::endl;
        int retries = 0;
        const int max_retries = 60;
        while (!shm.open() && retries < max_retries) {
            std::cout << "Attempt " << (retries + 1) << "/" << max_retries << ": shared memory not found, retrying..." << std::endl;
            Sleep(1000);
            retries++;
        }
        if (retries >= max_retries) {
            std::cerr << "Error: Failed to open shared memory" << std::endl;
            g_export_done = true;
            g_export_cv.notify_one();
            export_thread.join();
            return -1;
        }

        int frame_bias_pre = -2;
        int frame_syn_pre = -2;
        float avg_freq_delta = 0.0f;
        bool firstBatch = true;

        std::vector<double> batch_times;
        StageTimers total_timers;

        std::string dtmb_output_dir = "constellation_freq_realtime";
        MKDIR(dtmb_output_dir.c_str());

        std::cout << std::endl << "Starting real-time processing (3-Channel Optimized Pipeline Mode)..." << std::endl;
        std::cout << "Press Ctrl+C to stop processing" << std::endl << std::endl;

        int batch_count = 0;
        bool running = true;

        while (running) {
            auto batch_start = std::chrono::high_resolution_clock::now();

            StageTimers stage_timer;

            if (!shm.lock_rx()) {
                std::cerr << "Warning: lock_rx failed" << std::endl;
                continue;
            }

            if (shm.data()->is_running == 0) {
                std::cout << "Sender has stopped" << std::endl;
                running = false;
                shm.unlock_rx();
                break;
            }

            uint32_t batch_index = shm.data()->batch_index;
            uint64_t timestamp_us = shm.data()->timestamp_us;
            double timestamp_s = timestamp_us / 1000000.0;

            if (firstBatch)
            {
                dataMode = shm.data()->data_mode;
                std::cout << "Detected data mode: " << (dataMode == 0 ? "3-Channel Simulation (7.56 MHz)" : "8-Channel Measured (10 MHz)") << std::endl;
                
                if (dataMode == 1)
                {
                    originalSampleRate = 10.0f * 1e6f;
                    needsDownsampling = true;
                    inputBatchSize = MAX_INPUT_BATCH_SIZE;
                    
                    downsampleCache = create_fft_plan_cache(3, inputBatchSize, BATCH_SIZE);
                    if (!downsampleCache)
                    {
                        std::cerr << "Error: Failed to create downsample cache" << std::endl;
                        shm.unlock_rx();
                        g_export_done = true;
                        g_export_cv.notify_one();
                        export_thread.join();
                        return -1;
                    }
                    std::cout << "  Input sample rate: " << originalSampleRate / 1e6 << " MHz" << std::endl;
                    std::cout << "  Input batch size: " << inputBatchSize << " samples" << std::endl;
                    std::cout << "  Downsampling to: " << SAMPLE_RATE / 1e6 << " MHz" << std::endl;
                }
                else
                {
                    originalSampleRate = SAMPLE_RATE;
                    needsDownsampling = false;
                    inputBatchSize = BATCH_SIZE;
                    std::cout << "  Input sample rate: " << originalSampleRate / 1e6 << " MHz" << std::endl;
                }

                int shmDataSize = inputBatchSize * NUM_CHANNELS;
                CUDA_CHECK(cudaMallocHost(&h_pinnedData, shmDataSize * sizeof(cuFloatComplex)));
                std::cout << "Allocated pinned host buffer: " << shmDataSize << " elements" << std::endl;

                firstBatch = false;
            }

            std::cout << "Processing batch " << (batch_count + 1) << " (file index: " << batch_index
                      << ") - timestamp: " << timestamp_s << " s (" << timestamp_us << " us)" << std::endl;

            nvtxRangePushA("Stage: DataTransfer");
            auto t1 = std::chrono::high_resolution_clock::now();
            const std::complex<float>* shm_data = shm.data()->data;
            memcpy(h_pinnedData, shm_data, inputBatchSize * NUM_CHANNELS * sizeof(cuFloatComplex));

            std::vector<cuFloatComplex> h_ch0(inputBatchSize);
            std::vector<cuFloatComplex> h_ch1(inputBatchSize);
            std::vector<cuFloatComplex> h_ch2(inputBatchSize);
            for (int i = 0; i < inputBatchSize; i++) {
                h_ch0[i] = h_pinnedData[i * 3];
                h_ch1[i] = h_pinnedData[i * 3 + 1];
                h_ch2[i] = h_pinnedData[i * 3 + 2];
            }

            if (needsDownsampling) {
                nvtxRangePushA("Downsample");
                std::vector<cuFloatComplex> h_concat(3 * inputBatchSize);
                memcpy(h_concat.data(), h_ch0.data(), inputBatchSize * sizeof(cuFloatComplex));
                memcpy(h_concat.data() + inputBatchSize, h_ch1.data(), inputBatchSize * sizeof(cuFloatComplex));
                memcpy(h_concat.data() + 2 * inputBatchSize, h_ch2.data(), inputBatchSize * sizeof(cuFloatComplex));

                float* h_src = (float*)h_concat.data();
                float* dsResult = batch_descend_sample_gpu_with_cache(
                    h_src, 3, inputBatchSize, BATCH_SIZE, downsampleCache);

                if (dsResult) {
                    cufftComplex* dsComplex = (cufftComplex*)dsResult;
                    CUDA_CHECK(cudaMemcpyAsync(d_processedData, dsComplex,
                                               3 * BATCH_SIZE * sizeof(cufftComplex), cudaMemcpyHostToDevice, stream1));
                    CUDA_CHECK(cudaStreamSynchronize(stream1));
                    free(dsResult);
                }
                nvtxRangePop();
            } else {
                CUDA_CHECK(cudaMemcpyAsync(d_processedData, h_ch0.data(),
                                           BATCH_SIZE * sizeof(cufftComplex), cudaMemcpyHostToDevice, stream1));
                CUDA_CHECK(cudaMemcpyAsync(d_processedData + BATCH_SIZE, h_ch1.data(),
                                           BATCH_SIZE * sizeof(cufftComplex), cudaMemcpyHostToDevice, stream1));
                CUDA_CHECK(cudaMemcpyAsync(d_processedData + 2 * BATCH_SIZE, h_ch2.data(),
                                           BATCH_SIZE * sizeof(cufftComplex), cudaMemcpyHostToDevice, stream1));
                CUDA_CHECK(cudaStreamSynchronize(stream1));
            }

            auto t2 = std::chrono::high_resolution_clock::now();
            stage_timer.data_transfer = std::chrono::duration<double>(t2 - t1).count();
            nvtxRangePop();

            shm.unlock_rx();

            nvtxRangePushA("Stage: FrameSync");
            t1 = std::chrono::high_resolution_clock::now();
            int frame_bias_result = 0;
            int frame_syn_result = 0;
            dtmb_syn_device_with_cache(d_processedData, BATCH_SIZE, pn420,
                                       frame_bias_pre, frame_syn_pre,
                                       frame_bias_result, frame_syn_result, dtmbCache);
            t2 = std::chrono::high_resolution_clock::now();
            stage_timer.frame_sync = std::chrono::duration<double>(t2 - t1).count();
            nvtxRangePop();

            frame_bias_pre = frame_bias_result;
            frame_syn_pre = frame_syn_result;

            std::cout << "  Frame sync: bias=" << frame_bias_result << ", syn=" << frame_syn_result << std::endl;

            int freq_estimate_frames = 178;
            int freq_data_needed = (freq_estimate_frames + 1) * 4200;

            if (BATCH_SIZE >= frame_bias_result + freq_data_needed) {
                nvtxRangePushA("Stage: FreqEstimate");
                t1 = std::chrono::high_resolution_clock::now();
                cufftComplex* d_freq_data = d_processedData + frame_bias_result;

                FreqEstimateResult freq_result = dtmb_freq_estimate_gpu_device(
                    d_freq_data, BATCH_SIZE - frame_bias_result,
                    frame_syn_result, freq_estimate_frames, freqCache);
                t2 = std::chrono::high_resolution_clock::now();
                stage_timer.freq_estimate = std::chrono::duration<double>(t2 - t1).count();
                nvtxRangePop();

                avg_freq_delta = freq_result.freq_delta;

                fprintf(stdout, "  Method 0 (Used), Mean:%f, Std:%f, Max:%f, Min:%f\n",
                    freq_result.freq_delta, freq_result.freq_delta_std,
                    freq_result.freq_delta_max, freq_result.freq_delta_min);
                fprintf(stdout, "  Method 1 (Ref), Mean:%f, Std:%f, Max:%f, Min:%f\n",
                    freq_result.freq_delta1, freq_result.freq_delta1_std,
                    freq_result.freq_delta1_max, freq_result.freq_delta1_min);

                nvtxRangePushA("Stage: FreqCompensate");
                t1 = std::chrono::high_resolution_clock::now();

                freq_compensate_gpu_multi_channel_inplace(
                    d_processedData, avg_freq_delta, (float)signalParm.sample_rate,
                    outputChannels, BATCH_SIZE);
                t2 = std::chrono::high_resolution_clock::now();
                stage_timer.freq_compensate = std::chrono::duration<double>(t2 - t1).count();
                nvtxRangePop();

                int ref_clear_frames = 178;
                int ref_clear_data_needed = (ref_clear_frames + 1) * 4200;

                if (BATCH_SIZE >= frame_bias_result + ref_clear_data_needed) {
                    nvtxRangePushA("Stage: RefClear");
                    t1 = std::chrono::high_resolution_clock::now();

                    RefClearResultDevice ref_clear_result = ref_clear_gpu_batch_device(
                        (const cuFloatComplex*)d_processedData,
                        frame_bias_result,
                        ref_clear_data_needed,
                        frame_syn_result,
                        ref_clear_frames,
                        pn420_bz,
                        refClearCache);
                    t2 = std::chrono::high_resolution_clock::now();
                    stage_timer.ref_clear = std::chrono::duration<double>(t2 - t1).count();
                    nvtxRangePop();

                    std::cout << "  ref_clear completed (GPU-only path)" << std::endl;

                    nvtxRangePushA("Stage: ExportD2H");
                    t1 = std::chrono::high_resolution_clock::now();

                    ExportTask task;

                    size_t fft_freq_size = 3780 * ref_clear_frames;
                    std::vector<cuFloatComplex> h_data_in_fft_freq_gpu(fft_freq_size);
                    CUDA_CHECK(cudaMemcpy(h_data_in_fft_freq_gpu.data(), ref_clear_result.d_data_in_fft_freq,
                                          fft_freq_size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
                    task.data_in_fft_freq.resize(fft_freq_size);
                    for (size_t i = 0; i < fft_freq_size; i++) {
                        task.data_in_fft_freq[i] = std::complex<float>(
                            cuCrealf(h_data_in_fft_freq_gpu[i]), cuCimagf(h_data_in_fft_freq_gpu[i]));
                    }

                    size_t refclrdata_size = 4200 * ref_clear_frames;
                    std::vector<cuFloatComplex> h_refclrdata_gpu(refclrdata_size);
                    CUDA_CHECK(cudaMemcpy(h_refclrdata_gpu.data(), ref_clear_result.d_refclrdata,
                                          refclrdata_size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));

                    size_t ref_data_size = 3780 * ref_clear_frames;
                    task.ref_extracted.resize(ref_data_size);
                    for (int fi = 0; fi < ref_clear_frames; fi++) {
                        for (int i = 0; i < 3780; i++) {
                            cuFloatComplex val = h_refclrdata_gpu[fi * 4200 + 420 + i];
                            task.ref_extracted[fi * 3780 + i] = std::complex<float>(cuCrealf(val), cuCimagf(val));
                        }
                    }

                    std::vector<cuFloatComplex> compensated_ch1(ref_clear_data_needed);
                    std::vector<cuFloatComplex> compensated_ch2(ref_clear_data_needed);
                    CUDA_CHECK(cudaMemcpy(compensated_ch1.data(),
                                          d_processedData + BATCH_SIZE + frame_bias_result,
                                          ref_clear_data_needed * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(compensated_ch2.data(),
                                          d_processedData + 2 * BATCH_SIZE + frame_bias_result,
                                          ref_clear_data_needed * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));

                    int adjusted_bias = 4200 + 420 + ref_clear_result.delta;
                    task.monitor_ch1.resize(ref_data_size);
                    task.monitor_ch2.resize(ref_data_size);
                    for (int fi = 0; fi < ref_clear_frames; fi++) {
                        for (int i = 0; i < 3780; i++) {
                            int local_idx = adjusted_bias + fi * 4200 + i;
                            task.monitor_ch1[fi * 3780 + i] = std::complex<float>(
                                cuCrealf(compensated_ch1[local_idx]), cuCimagf(compensated_ch1[local_idx]));
                            task.monitor_ch2[fi * 3780 + i] = std::complex<float>(
                                cuCrealf(compensated_ch2[local_idx]), cuCimagf(compensated_ch2[local_idx]));
                        }
                    }

                    char batch_filename[256];
                    snprintf(batch_filename, sizeof(batch_filename), "%s/constellation_batch_%03d.bin",
                             dtmb_output_dir.c_str(), batch_index);
                    task.constellation_filename = batch_filename;

                    char dtmb_result_filename[256];
                    snprintf(dtmb_result_filename, sizeof(dtmb_result_filename),
                             "%s/dtmb_processing_results_%03d.bin", dtmb_output_dir.c_str(), batch_index);
                    task.dtmb_result_filename = dtmb_result_filename;

                    t2 = std::chrono::high_resolution_clock::now();
                    stage_timer.export_d2h = std::chrono::duration<double>(t2 - t1).count();
                    nvtxRangePop();

                    {
                        std::lock_guard<std::mutex> lock(g_export_mutex);
                        g_export_queue.push(std::move(task));
                    }
                    g_export_cv.notify_one();

                    std::cout << "  DTMB export queued (async)" << std::endl;
                }
            }

            auto batch_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> batch_elapsed = batch_end - batch_start;
            double batch_time = batch_elapsed.count();
            batch_times.push_back(batch_time);

            total_timers.data_transfer += stage_timer.data_transfer;
            total_timers.frame_sync += stage_timer.frame_sync;
            total_timers.freq_estimate += stage_timer.freq_estimate;
            total_timers.freq_compensate += stage_timer.freq_compensate;
            total_timers.ref_clear += stage_timer.ref_clear;
            total_timers.export_d2h += stage_timer.export_d2h;

            std::cout << "Batch time: " << batch_time << " s" << std::endl << std::endl;

            batch_count++;
        }

        g_export_done = true;
        g_export_cv.notify_one();
        export_thread.join();

        if (batch_times.size() >= 1) {
            double total_time = 0.0;
            for (size_t i = 0; i < batch_times.size(); i++) {
                total_time += batch_times[i];
            }
            double avg_time = total_time / batch_times.size();

            std::cout << "\n========== Timing Statistics ==========" << std::endl;
            std::cout << "Average batch time (" << batch_times.size() << " batches): " << avg_time << " seconds" << std::endl;
            std::cout << "\n--- Stage Breakdown ---" << std::endl;

            int count = (int)batch_times.size();
            double avg_data_transfer = total_timers.data_transfer / count;
            double avg_frame_sync = total_timers.frame_sync / count;
            double avg_freq_estimate = total_timers.freq_estimate / count;
            double avg_freq_compensate = total_timers.freq_compensate / count;
            double avg_ref_clear = total_timers.ref_clear / count;
            double avg_export_d2h = total_timers.export_d2h / count;

            std::cout << "  Data transfer:   " << avg_data_transfer << " s ("
                      << (avg_data_transfer / avg_time * 100) << "%)" << std::endl;
            std::cout << "  Frame sync:      " << avg_frame_sync << " s ("
                      << (avg_frame_sync / avg_time * 100) << "%)" << std::endl;
            std::cout << "  Freq estimate:   " << avg_freq_estimate << " s ("
                      << (avg_freq_estimate / avg_time * 100) << "%)" << std::endl;
            std::cout << "  Freq compensate: " << avg_freq_compensate << " s ("
                      << (avg_freq_compensate / avg_time * 100) << "%)" << std::endl;
            std::cout << "  Ref clear:       " << avg_ref_clear << " s ("
                      << (avg_ref_clear / avg_time * 100) << "%)" << std::endl;
            std::cout << "  Export D2H:      " << avg_export_d2h << " s ("
                      << (avg_export_d2h / avg_time * 100) << "%)" << std::endl;
            std::cout << "  (File I/O is async and not included in batch time)" << std::endl;
            std::cout << "========================================" << std::endl;
        }

        CUDA_CHECK(cudaStreamDestroy(stream1));
        CUDA_CHECK(cudaFreeHost(h_pinnedData));
        CUDA_CHECK(cudaFree(d_processedData));
        if (downsampleCache) destroy_fft_plan_cache(downsampleCache);
        destroy_dtmb_gpu_cache(dtmbCache);
        destroy_freq_estimate_gpu_cache(freqCache);
        destroy_ref_clear_gpu_cache(refClearCache, true);
        destroy_freq_compensate_gpu_cache(freqCompCache);
        std::cout << "========== finished ==========" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        g_export_done = true;
        g_export_cv.notify_one();
        return -1;
    }

    return 0;
}
