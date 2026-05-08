// dtmb_generator.cpp - DTMB信号发生器
// 根据GB 20600-2006标准生成DTMB信号
// 帧结构: 420点帧头(PN420) + 3780点帧体 = 4200点/帧

#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <random>
#include <cmath>
#include "pn420_data.h"

using namespace std;

typedef complex<float> cf32;

// 4QAM星座表
const cf32 qam4_table[4] = {
    cf32(1.0f, 1.0f) / sqrtf(2.0f),
    cf32(1.0f, -1.0f) / sqrtf(2.0f),
    cf32(-1.0f, 1.0f) / sqrtf(2.0f),
    cf32(-1.0f, -1.0f) / sqrtf(2.0f)
};

// 简单的IFFT实现（为了演示，实际可用FFTW或cuFFT）
void ifft(vector<cf32>& data) {
    int n = data.size();
    vector<cf32> result(n);
    
    for (int k = 0; k < n; k++) {
        cf32 sum(0, 0);
        for (int m = 0; m < n; m++) {
            float angle = 2.0f * M_PI * k * m / n;
            cf32 exp_term(cosf(angle), sinf(angle));
            sum += data[m] * exp_term;
        }
        result[k] = sum / (float)n;
    }
    data = result;
}

vector<cf32> generate_4qam_symbols(int num_symbols, mt19937& rng) {
    vector<cf32> symbols(num_symbols);
    uniform_int_distribution<int> dist(0, 3);
    
    for (int i = 0; i < num_symbols; i++) {
        symbols[i] = qam4_table[dist(rng)];
    }
    return symbols;
}

vector<cf32> get_pn420_header(int frame_idx) {
    vector<cf32> header(420);
    
    // 使用1-225循环的PN420序列
    int pn_col = frame_idx % 225;
    
    for (int i = 0; i < 420; i++) {
        // 从pn420_01获取原始0/1值，转换为(1-2*bit)*(1+1j)/sqrt(2)
        unsigned char bit = pn420_01[i][pn_col];
        float val = 1.0f - 2.0f * (float)bit;
        header[i] = cf32(val, val) / sqrtf(2.0f) * sqrtf(2.0f);
    }
    
    return header;
}

int main(int argc, char* argv[]) {
    string output_filename = "dtmb_test_signal_180frames_cpp.bin";
    int num_frames = 180;
    
    if (argc >= 2) {
        output_filename = argv[1];
    }
    if (argc >= 3) {
        num_frames = stoi(argv[2]);
    }
    
    cout << "========================================" << endl;
    cout << "  DTMB 信号发生器 (C++版本)" << endl;
    cout << "========================================" << endl << endl;
    
    cout << "输出文件: " << output_filename << endl;
    cout << "生成帧数: " << num_frames << endl;
    
    // 随机数生成器
    mt19937 rng(12345);
    
    // 总信号
    int frame_length = 4200;
    int total_samples = num_frames * frame_length;
    vector<cf32> dtmb_signal(total_samples);
    
    // 生成每一帧
    for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        cout << "正在生成第 " << (frame_idx + 1) << " / " << num_frames << " 帧..." << endl;
        
        // 1. 帧头 - PN420序列
        vector<cf32> header = get_pn420_header(frame_idx);
        
        // 2. 帧体 - 4QAM调制 + IFFT
        vector<cf32> freq_symbols = generate_4qam_symbols(3780, rng);
        ifft(freq_symbols);
        
        // 3. 组合到总信号
        int frame_start = frame_idx * frame_length;
        for (int i = 0; i < 420; i++) {
            dtmb_signal[frame_start + i] = header[i];
        }
        for (int i = 0; i < 3780; i++) {
            dtmb_signal[frame_start + 420 + i] = freq_symbols[i];
        }
    }
    
    // 导出为二进制文件
    ofstream file(output_filename, ios::binary);
    if (!file) {
        cerr << "无法创建文件: " << output_filename << endl;
        return 1;
    }
    
    for (const auto& sample : dtmb_signal) {
        float real_part = sample.real();
        float imag_part = sample.imag();
        file.write(reinterpret_cast<const char*>(&real_part), sizeof(float));
        file.write(reinterpret_cast<const char*>(&imag_part), sizeof(float));
    }
    
    file.close();
    
    cout << endl;
    cout << "========================================" << endl;
    cout << "  信号生成完成！" << endl;
    cout << "========================================" << endl;
    cout << "总采样点: " << total_samples << endl;
    
    return 0;
}
