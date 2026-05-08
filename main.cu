#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <sys/stat.h>
#include <sys/types.h>

#include <cuda_runtime.h>
#include <cufft.h>
#include "read_dtmb_signal.h"
#include "dtmb_frame_sync.h"
#include "dtmb_freq_estimate.h"
#include "ref_clear.h"

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
    double data_load;
    double frame_sync;
    double freq_estimate;
    double freq_compensate;
    double ref_clear;
    double other;

    StageTimers() : data_load(0), frame_sync(0), freq_estimate(0), freq_compensate(0), ref_clear(0), other(0) {}
};

int main() {
    std::string filename = "D:/CUDA_Program/DTMB_process/DTMB_Sender_GUI/dtmb_time_signal_triple.bin";

    try {
        std::vector<std::complex<float>> pn420;
        if (!read_pn420("pn420_0.mat", pn420)) {
            std::cerr << "Error: Failed to read PN420 sequence" << std::endl;
            return -1;
        }

        std::vector<std::complex<float>> signalDataCh0, signalDataCh1, signalDataCh2;
        if (!read_dtmb_signal_binary_triple(filename, signalDataCh0, signalDataCh1, signalDataCh2)) {
            std::cerr << "Error: Failed to read DTMB triple signal file" << std::endl;
            return -1;
        }

        std::cout << "DTMB triple signal file loaded: " << signalDataCh0.size() << " samples per channel" << std::endl;

        int numBatches = signalDataCh0.size() / BATCH_SIZE;
        std::cout << "Number of batches to process: " << numBatches << std::endl;

        int outputChannels = 3;
        HEstimateImprovedGPUCache* hEstImprovedCache = create_h_estimate_improved_gpu_cache_with_batch(256);
        DTMBGPUCache* dtmbCache = create_dtmb_gpu_cache(BATCH_SIZE, 3 * 4200);
        FreqEstimateGPUCache* freqCache = create_freq_estimate_gpu_cache(256);
        RefClearGPUCache* refClearCache = create_ref_clear_gpu_cache(256, hEstImprovedCache);
        FreqCompensateGPUCache* freqCompCache = create_freq_compensate_gpu_cache(outputChannels * BATCH_SIZE);

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

        int frame_bias_pre = -2;
        int frame_syn_pre = -2;
        float avg_freq_delta = 0.0f;

        std::vector<double> batch_times;
        StageTimers total_timers;

        std::vector<std::complex<float>> ref_clear_input;

        std::string dtmb_output_dir = "D:/CUDA_Program/CudaRuntime3/dtmb_processing_results";
        MKDIR(dtmb_output_dir.c_str());

        std::string output_dir = "constellation_freq";
        MKDIR(output_dir.c_str());

        std::cout << std::endl << "Starting batch processing (3-Channel GPU Pipeline Mode)..." << std::endl << std::endl;

        for (int batch = 0; batch < numBatches; batch++) {
            auto batch_start = std::chrono::high_resolution_clock::now();

            std::cout << "Processing batch " << batch + 1 << "/" << numBatches << "..." << std::endl;

            StageTimers stage_timer;

            auto t1 = std::chrono::high_resolution_clock::now();
            int startIdx = batch * BATCH_SIZE;

            std::vector<std::complex<float>> batchDataCh0(signalDataCh0.begin() + startIdx, signalDataCh0.begin() + startIdx + BATCH_SIZE);
            std::vector<std::complex<float>> batchDataCh1(signalDataCh1.begin() + startIdx, signalDataCh1.begin() + startIdx + BATCH_SIZE);
            std::vector<std::complex<float>> batchDataCh2(signalDataCh2.begin() + startIdx, signalDataCh2.begin() + startIdx + BATCH_SIZE);

            CUDA_CHECK(cudaMemcpyAsync(d_processedData, batchDataCh0.data(),
                                       BATCH_SIZE * sizeof(cufftComplex), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpyAsync(d_processedData + BATCH_SIZE, batchDataCh1.data(),
                                       BATCH_SIZE * sizeof(cufftComplex), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpyAsync(d_processedData + 2 * BATCH_SIZE, batchDataCh2.data(),
                                       BATCH_SIZE * sizeof(cufftComplex), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());

            auto t2 = std::chrono::high_resolution_clock::now();
            stage_timer.data_load = std::chrono::duration<double>(t2 - t1).count();

            t1 = std::chrono::high_resolution_clock::now();
            int frame_bias_result = 0;
            int frame_syn_result = 0;
            dtmb_syn_device_with_cache(d_processedData, BATCH_SIZE, pn420,
                                       frame_bias_pre, frame_syn_pre,
                                       frame_bias_result, frame_syn_result, dtmbCache);
            t2 = std::chrono::high_resolution_clock::now();
            stage_timer.frame_sync = std::chrono::duration<double>(t2 - t1).count();

            frame_bias_pre = frame_bias_result;
            frame_syn_pre = frame_syn_result;

            std::cout << "  Frame sync: bias=" << frame_bias_result << ", syn=" << frame_syn_result << std::endl;

            int freq_estimate_frames = 178;
            int freq_data_needed = (freq_estimate_frames + 1) * 4200;

            if (frame_bias_result >= 0 && frame_syn_result >= 1 && 
                frame_syn_result <= 225 && BATCH_SIZE >= frame_bias_result + freq_data_needed) {
                t1 = std::chrono::high_resolution_clock::now();
                cufftComplex* d_freq_data = d_processedData + frame_bias_result;

                FreqEstimateResult freq_result = dtmb_freq_estimate_gpu_device(
                    d_freq_data, BATCH_SIZE - frame_bias_result,
                    frame_syn_result, freq_estimate_frames, freqCache);
                t2 = std::chrono::high_resolution_clock::now();
                stage_timer.freq_estimate = std::chrono::duration<double>(t2 - t1).count();

                avg_freq_delta = freq_result.freq_delta;

                fprintf(stdout, "  Method 0 (Used), Mean:%f, Std:%f, Max:%f, Min:%f\n",
                    freq_result.freq_delta, freq_result.freq_delta_std,
                    freq_result.freq_delta_max, freq_result.freq_delta_min);
                fprintf(stdout, "  Method 1 (Ref), Mean:%f, Std:%f, Max:%f, Min:%f\n",
                    freq_result.freq_delta1, freq_result.freq_delta1_std,
                    freq_result.freq_delta1_max, freq_result.freq_delta1_min);

                t1 = std::chrono::high_resolution_clock::now();
                freq_compensate_gpu_multi_channel_inplace(
                    d_processedData, avg_freq_delta, (float)SAMPLE_RATE,
                    outputChannels, BATCH_SIZE);
                t2 = std::chrono::high_resolution_clock::now();
                stage_timer.freq_compensate = std::chrono::duration<double>(t2 - t1).count();

                int ref_clear_frames = 178;
                int ref_clear_data_needed = (ref_clear_frames + 1) * 4200;

                if (frame_bias_result >= 0 && frame_syn_result >= 1 && 
                    frame_syn_result <= 225 && BATCH_SIZE >= frame_bias_result + ref_clear_data_needed) {
                    t1 = std::chrono::high_resolution_clock::now();

                    std::vector<std::complex<float>> compensated_data_three_channel(ref_clear_data_needed * 3);
                    for (int ch = 0; ch < 3; ch++) {
                        CUDA_CHECK(cudaMemcpy(compensated_data_three_channel.data() + ch * ref_clear_data_needed,
                                              d_processedData + ch * BATCH_SIZE + frame_bias_result,
                                              ref_clear_data_needed * sizeof(cufftComplex),
                                              cudaMemcpyDeviceToHost));
                    }

                    ref_clear_input.resize(ref_clear_data_needed);
                    for (int s = 0; s < ref_clear_data_needed; s++) {
                        ref_clear_input[s] = std::complex<float>(
                            compensated_data_three_channel[s].real(),
                            compensated_data_three_channel[s].imag()
                        );
                    }

                    RefClearResult ref_clear_result = ref_clear_gpu_batch(
                        ref_clear_input,
                        frame_syn_result,
                        ref_clear_frames,
                        pn420_bz,
                        refClearCache);
                    t2 = std::chrono::high_resolution_clock::now();
                    stage_timer.ref_clear = std::chrono::duration<double>(t2 - t1).count();

                    std::cout << "  ref_clear completed" << std::endl;

                    char dtmb_result_filename[256];
                    snprintf(dtmb_result_filename, sizeof(dtmb_result_filename),
                             "%s/dtmb_processing_results_%03d.bin", dtmb_output_dir.c_str(), batch);
                    std::ofstream dtmb_file(dtmb_result_filename, std::ios::binary);
                    if (dtmb_file.is_open()) {
                        size_t ref_data_size = 3780 * ref_clear_frames;
                        std::vector<std::complex<float>> ref_extracted(ref_data_size);

                        for (int fi = 0; fi < ref_clear_frames; fi++) {
                            for (int i = 0; i < 3780; i++) {
                                ref_extracted[fi * 3780 + i] = ref_clear_result.refclrdata[fi * 4200 + 420 + i];
                            }
                        }

                        size_t monitor_data_size = 3780 * ref_clear_frames;
                        std::vector<std::complex<float>> monitor_channel1_full(monitor_data_size);
                        std::vector<std::complex<float>> monitor_channel2_full(monitor_data_size);

                        int adjusted_bias = 4200 + 420 + ref_clear_result.delta;

                        for (int fi = 0; fi < ref_clear_frames; fi++) {
                            for (int i = 0; i < 3780; i++) {
                                int local_idx = adjusted_bias + fi * 4200 + i;
                                monitor_channel1_full[fi * 3780 + i] = compensated_data_three_channel[local_idx + ref_clear_data_needed];
                                monitor_channel2_full[fi * 3780 + i] = compensated_data_three_channel[local_idx + 2 * ref_clear_data_needed];
                            }
                        }

                        dtmb_file.write(reinterpret_cast<const char*>(ref_extracted.data()),
                                       ref_extracted.size() * sizeof(std::complex<float>));

                        dtmb_file.write(reinterpret_cast<const char*>(monitor_channel1_full.data()),
                                       monitor_channel1_full.size() * sizeof(std::complex<float>));

                        dtmb_file.write(reinterpret_cast<const char*>(monitor_channel2_full.data()),
                                       monitor_channel2_full.size() * sizeof(std::complex<float>));

                        dtmb_file.close();
                        std::cout << "  DTMB processing results exported to " << dtmb_result_filename << std::endl;
                        std::cout << "    Ref: " << ref_extracted.size() << ", Sur1: " << monitor_channel1_full.size()
                                  << ", Sur2: " << monitor_channel2_full.size() << " samples" << std::endl;
                    }

                    char batch_filename[256];
                    snprintf(batch_filename, sizeof(batch_filename), "%s/constellation_batch_%03d.bin",
                             output_dir.c_str(), batch);
                    export_constellation_data_binary(
                        ref_clear_result.data_in_fft_freq,
                        batch_filename);
                }
            }

            auto batch_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> batch_elapsed = batch_end - batch_start;
            double batch_time = batch_elapsed.count();
            batch_times.push_back(batch_time);

            total_timers.data_load += stage_timer.data_load;
            total_timers.frame_sync += stage_timer.frame_sync;
            total_timers.freq_estimate += stage_timer.freq_estimate;
            total_timers.freq_compensate += stage_timer.freq_compensate;
            total_timers.ref_clear += stage_timer.ref_clear;

            std::cout << "Batch time: " << batch_time << "s" << std::endl << std::endl;
        }

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
            double avg_data_load = total_timers.data_load / count;
            double avg_frame_sync = total_timers.frame_sync / count;
            double avg_freq_estimate = total_timers.freq_estimate / count;
            double avg_freq_compensate = total_timers.freq_compensate / count;
            double avg_ref_clear = total_timers.ref_clear / count;

            std::cout << "  Data load:       " << avg_data_load << " s ("
                      << (avg_data_load/avg_time*100) << "%)" << std::endl;
            std::cout << "  Frame sync:      " << avg_frame_sync << " s ("
                      << (avg_frame_sync/avg_time*100) << "%)" << std::endl;
            std::cout << "  Freq estimate:   " << avg_freq_estimate << " s ("
                      << (avg_freq_estimate/avg_time*100) << "%)" << std::endl;
            std::cout << "  Freq compensate: " << avg_freq_compensate << " s ("
                      << (avg_freq_compensate/avg_time*100) << "%)" << std::endl;
            std::cout << "  Ref clear:       " << avg_ref_clear << " s ("
                      << (avg_ref_clear/avg_time*100) << "%)" << std::endl;
            std::cout << "========================================" << std::endl;
        }

        CUDA_CHECK(cudaFree(d_processedData));
        destroy_dtmb_gpu_cache(dtmbCache);
        destroy_freq_estimate_gpu_cache(freqCache);
        destroy_ref_clear_gpu_cache(refClearCache, true);
        destroy_freq_compensate_gpu_cache(freqCompCache);
        std::cout << "========== finished ==========" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
