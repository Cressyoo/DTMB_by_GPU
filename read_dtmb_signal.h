#ifndef READ_DTMB_SIGNAL_H
#define READ_DTMB_SIGNAL_H

#include <string>
#include <vector>
#include <complex>
#include <cstdint>

struct DTMBSignalParm {
    int32_t frame_header_len;
    int32_t frame_body_len;
    int32_t frame_len;
    int32_t num_frames;
    int32_t fft_size;
    int32_t total_samples;
    int32_t num_channels;
    float sample_rate;
};

DTMBSignalParm get_dtmb_signal_param();

bool read_dtmb_signal_binary(const std::string& filename, std::vector<std::complex<float>>& signal_data, int channel_idx = 0);

bool read_dtmb_signal_binary_dual(const std::string& filename,
                                    std::vector<std::complex<float>>& signal_data_ch0,
                                    std::vector<std::complex<float>>& signal_data_ch1);

bool read_dtmb_signal_binary_triple(const std::string& filename,
                                      std::vector<std::complex<float>>& signal_data_ch0,
                                      std::vector<std::complex<float>>& signal_data_ch1,
                                      std::vector<std::complex<float>>& signal_data_ch2);

#endif // READ_DTMB_SIGNAL_H
