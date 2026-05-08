#include "read_dtmb_signal.h"
#include <fstream>
#include <iostream>
#include <cmath>

DTMBSignalParm get_dtmb_signal_param() {
    DTMBSignalParm parm;
    parm.frame_header_len = 420;
    parm.frame_body_len = 3780;
    parm.frame_len = 420 + 3780;
    parm.num_frames = 9000;
    parm.fft_size = 3780;
    parm.num_channels = 3;
    parm.sample_rate = 7.56e6f;
    parm.total_samples = parm.frame_len * parm.num_frames;
    return parm;
}

bool read_dtmb_signal_binary(const std::string& filename, std::vector<std::complex<float>>& signal_data, int channel_idx) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open file: " << filename << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    DTMBSignalParm parm = get_dtmb_signal_param();
    int num_channels = parm.num_channels;
    std::streamsize expected_size = static_cast<std::streamsize>(parm.total_samples * num_channels * 2 * sizeof(float));

    if (file_size < expected_size) {
        std::cerr << "[WARNING] File size " << file_size << " bytes is less than expected " << expected_size << " bytes" << std::endl;
        int num_samples = static_cast<int>(file_size / (num_channels * 2 * sizeof(float)));
        parm.num_frames = num_samples / parm.frame_len;
        parm.total_samples = parm.num_frames * parm.frame_len;
        std::cout << "[INFO] Adjusting to " << parm.num_frames << " frames, " << parm.total_samples << " samples" << std::endl;
    }

    signal_data.resize(parm.total_samples);

    float real_part, imag_part;
    for (int i = 0; i < parm.total_samples; i++) {
        for (int ch = 0; ch < num_channels; ch++) {
            file.read(reinterpret_cast<char*>(&real_part), sizeof(float));
            file.read(reinterpret_cast<char*>(&imag_part), sizeof(float));
            if (!file) {
                std::cerr << "[ERROR] Failed to read sample " << i << " channel " << ch << std::endl;
                signal_data.resize(i);
                return false;
            }
            if (ch == channel_idx) {
                signal_data[i] = std::complex<float>(real_part, imag_part);
            }
        }
    }

    file.close();
    std::cout << "[INFO] Loaded channel " << channel_idx << ": " << parm.num_frames << " frames, " << parm.total_samples << " samples from " << filename << std::endl;
    return true;
}

bool read_dtmb_signal_binary_dual(const std::string& filename,
                                    std::vector<std::complex<float>>& signal_data_ch0,
                                    std::vector<std::complex<float>>& signal_data_ch1) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open file: " << filename << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    DTMBSignalParm parm = get_dtmb_signal_param();
    int num_channels = parm.num_channels;
    std::streamsize expected_size = static_cast<std::streamsize>(parm.total_samples * num_channels * 2 * sizeof(float));

    if (file_size < expected_size) {
        std::cerr << "[WARNING] File size " << file_size << " bytes is less than expected " << expected_size << " bytes" << std::endl;
        int num_samples = static_cast<int>(file_size / (num_channels * 2 * sizeof(float)));
        parm.num_frames = num_samples / parm.frame_len;
        parm.total_samples = parm.num_frames * parm.frame_len;
        std::cout << "[INFO] Adjusting to " << parm.num_frames << " frames, " << parm.total_samples << " samples" << std::endl;
    }

    signal_data_ch0.resize(parm.total_samples);
    signal_data_ch1.resize(parm.total_samples);

    float real_part, imag_part;
    for (int i = 0; i < parm.total_samples; i++) {
        for (int ch = 0; ch < num_channels; ch++) {
            file.read(reinterpret_cast<char*>(&real_part), sizeof(float));
            file.read(reinterpret_cast<char*>(&imag_part), sizeof(float));
            if (!file) {
                std::cerr << "[ERROR] Failed to read sample " << i << " channel " << ch << std::endl;
                signal_data_ch0.resize(i);
                signal_data_ch1.resize(i);
                return false;
            }
            if (ch == 0) {
                signal_data_ch0[i] = std::complex<float>(real_part, imag_part);
            } else if (ch == 1) {
                signal_data_ch1[i] = std::complex<float>(real_part, imag_part);
            }
        }
    }

    file.close();
    std::cout << "[INFO] Loaded dual channels: " << parm.num_frames << " frames, " << parm.total_samples << " samples each from " << filename << std::endl;
    return true;
}

bool read_dtmb_signal_binary_triple(const std::string& filename,
                                      std::vector<std::complex<float>>& signal_data_ch0,
                                      std::vector<std::complex<float>>& signal_data_ch1,
                                      std::vector<std::complex<float>>& signal_data_ch2) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open file: " << filename << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    DTMBSignalParm parm = get_dtmb_signal_param();
    int num_channels = parm.num_channels;
    std::streamsize expected_size = static_cast<std::streamsize>(parm.total_samples * num_channels * 2 * sizeof(float));

    if (file_size < expected_size) {
        std::cerr << "[WARNING] File size " << file_size << " bytes is less than expected " << expected_size << " bytes" << std::endl;
        int num_samples = static_cast<int>(file_size / (num_channels * 2 * sizeof(float)));
        parm.num_frames = num_samples / parm.frame_len;
        parm.total_samples = parm.num_frames * parm.frame_len;
        std::cout << "[INFO] Adjusting to " << parm.num_frames << " frames, " << parm.total_samples << " samples" << std::endl;
    }

    signal_data_ch0.resize(parm.total_samples);
    signal_data_ch1.resize(parm.total_samples);
    signal_data_ch2.resize(parm.total_samples);

    float real_part, imag_part;
    for (int i = 0; i < parm.total_samples; i++) {
        for (int ch = 0; ch < num_channels; ch++) {
            file.read(reinterpret_cast<char*>(&real_part), sizeof(float));
            file.read(reinterpret_cast<char*>(&imag_part), sizeof(float));
            if (!file) {
                std::cerr << "[ERROR] Failed to read sample " << i << " channel " << ch << std::endl;
                signal_data_ch0.resize(i);
                signal_data_ch1.resize(i);
                signal_data_ch2.resize(i);
                return false;
            }
            if (ch == 0) {
                signal_data_ch0[i] = std::complex<float>(real_part, imag_part);
            } else if (ch == 1) {
                signal_data_ch1[i] = std::complex<float>(real_part, imag_part);
            } else if (ch == 2) {
                signal_data_ch2[i] = std::complex<float>(real_part, imag_part);
            }
        }
    }

    file.close();
    std::cout << "[INFO] Loaded triple channels: " << parm.num_frames << " frames, " << parm.total_samples << " samples each from " << filename << std::endl;
    return true;
}
