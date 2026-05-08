#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <complex>
#include <cmath>
#include <cstdint>


// 结构体定义，对应MATLAB的SysParm
struct SysParm {
    int32_t VersionNo;
    int32_t FileHeadLen;
    std::string SignalType;
    std::string SignalPolarity;
    double f0;                      // 中心频率 (Hz)
    std::string SignalMode;         // 16字符
    std::string RxTxID;             // 32字符
    std::vector<double> RxPos;      // 3个元素 [x,y,z]
    std::vector<double> TxPos;      // 3个元素 [x,y,z]
    std::string ArrayType;
    float ThetaNormal;              // 阵列法向(度)
    std::string ArrayPolarity;
    int32_t ArrayDimension;
    std::vector<float> AntCord;     // ArrayDimension * 3，行主序：每个阵元3个坐标连续
    std::string CaliMethod;
    float CaliStepAngle;
    std::vector<std::complex<float>> CaliMatrix; // ArrayDimension * numAngles，行主序：每行对应一个阵元
    float fs;                        // 采样率 (Hz)
    double StartSampleTime;           // UNIX时间戳
    uint64_t DatLen;                  // 数据体长度
    int32_t ChanNum;
    std::vector<float> Att;           // 2 * ChanNum，列主序：先所有通道的第1个衰减，再所有通道的第2个衰减
};

// 辅助函数：将整数信号类型转换为字符串
std::string signalTypeToString(int32_t type) {
    switch (type) {
    case -1: return "Unknown";
    case 0:  return "DTMB";
    case 1:  return "CMMB";
    case 2:  return "FM";
    default: return "Unknown";
    }
}

// 辅助函数：将整数极化转换为字符串
std::string polarityToString(int32_t pol) {
    switch (pol) {
    case -1: return "Unknown";
    case 0:  return "V";   // 垂直
    case 1:  return "H";   // 水平
    default: return "Unknown";
    }
}

// 辅助函数：将整数阵列类型转换为字符串
std::string arrayTypeToString(int32_t type) {
    switch (type) {
    case 0:  return "圆阵";
    case 1:  return "线阵";
    default: return "Unknown";
    }
}

// 辅助函数：将整数校准方法转换为字符串
std::string caliMethodToString(int32_t method) {
    switch (method) {
    case -1: return "无校准值";
    case 0:  return "解算法";
    case 1:  return "插值法";
    case 2:  return "解算法和插值法";
    default: return "Unknown";
    }
}

// 主函数：读取文件头并填充SysParm结构体
SysParm func_ReadSrcFileHead(const std::string& FilePath) {
    SysParm sp;
    std::ifstream file(FilePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + FilePath);
    }

    // 1. 读取固定长度标量
    file.read(reinterpret_cast<char*>(&sp.VersionNo), sizeof(sp.VersionNo));
    file.read(reinterpret_cast<char*>(&sp.FileHeadLen), sizeof(sp.FileHeadLen));

    int32_t tmpSignalType, tmpSignalPolarity;
    file.read(reinterpret_cast<char*>(&tmpSignalType), sizeof(tmpSignalType));
    file.read(reinterpret_cast<char*>(&tmpSignalPolarity), sizeof(tmpSignalPolarity));

    float f0_mhz;
    file.read(reinterpret_cast<char*>(&f0_mhz), sizeof(f0_mhz));
    sp.f0 = static_cast<double>(f0_mhz) * 1e6;

    // 2. 读取字符数组
    char signalModeBuf[16];
    file.read(signalModeBuf, 16);
    sp.SignalMode.assign(signalModeBuf, 16);

    char rxTxIDBuf[32];
    file.read(rxTxIDBuf, 32);
    sp.RxTxID.assign(rxTxIDBuf, 32);

    // 3. 读取位置信息（双精度数组）
    double rxPosArr[3], txPosArr[3];
    file.read(reinterpret_cast<char*>(rxPosArr), 3 * sizeof(double));
    file.read(reinterpret_cast<char*>(txPosArr), 3 * sizeof(double));
    sp.RxPos.assign(rxPosArr, rxPosArr + 3);
    sp.TxPos.assign(txPosArr, txPosArr + 3);

    // 4. 读取阵列相关
    int32_t tmpArrayType, tmpArrayPolarity;
    file.read(reinterpret_cast<char*>(&tmpArrayType), sizeof(tmpArrayType));
    file.read(reinterpret_cast<char*>(&sp.ThetaNormal), sizeof(sp.ThetaNormal));
    file.read(reinterpret_cast<char*>(&tmpArrayPolarity), sizeof(tmpArrayPolarity));
    file.read(reinterpret_cast<char*>(&sp.ArrayDimension), sizeof(sp.ArrayDimension));

    // 5. 读取天线坐标
    int antCordSize = sp.ArrayDimension * 3;
    sp.AntCord.resize(antCordSize);
    file.read(reinterpret_cast<char*>(sp.AntCord.data()), antCordSize * sizeof(float));

    // 6. 读取校准相关
    int32_t tmpCaliMethod;
    file.read(reinterpret_cast<char*>(&tmpCaliMethod), sizeof(tmpCaliMethod));
    file.read(reinterpret_cast<char*>(&sp.CaliStepAngle), sizeof(sp.CaliStepAngle));

    // 计算角度个数（四舍五入取整）
    int numAngles = static_cast<int>(std::round(360.0f / sp.CaliStepAngle));
    int totalComplex = sp.ArrayDimension * numAngles;
    int totalFloat = totalComplex * 2;  // 每个复数两个float

    std::vector<float> caliRaw(totalFloat);
    file.read(reinterpret_cast<char*>(caliRaw.data()), totalFloat * sizeof(float));

    // 将原始数据转换为复数矩阵（行主序：每个阵元一行，每列一个角度）
    sp.CaliMatrix.resize(totalComplex);
    // 原始数据存储顺序：按角度组，每个角度内按阵元交错实虚
    // 即：角度1：实部(阵元1),虚部(阵元1),实部(阵元2),虚部(阵元2),...
    // 我们要填充为行主序：CaliMatrix[阵元][角度]
    for (int a = 0; a < numAngles; ++a) {
        for (int e = 0; e < sp.ArrayDimension; ++e) {
            int rawIdx = (a * sp.ArrayDimension + e) * 2;  // 每个复数两个float
            float real = caliRaw[rawIdx];
            float imag = caliRaw[rawIdx + 1];
            int matIdx = e * numAngles + a;               // 行主序索引
            sp.CaliMatrix[matIdx] = std::complex<float>(real, imag);
        }
    }

    // 7. 读取采样率、时间戳、数据长度、通道数
    file.read(reinterpret_cast<char*>(&sp.fs), sizeof(sp.fs));
    file.read(reinterpret_cast<char*>(&sp.StartSampleTime), sizeof(sp.StartSampleTime));
    file.read(reinterpret_cast<char*>(&sp.DatLen), sizeof(sp.DatLen));
    file.read(reinterpret_cast<char*>(&sp.ChanNum), sizeof(sp.ChanNum));

    // 8. 读取衰减数据
    int attSize = 2 * sp.ChanNum;
    sp.Att.resize(attSize);
    file.read(reinterpret_cast<char*>(sp.Att.data()), attSize * sizeof(float));

    // 9. 转换为字符串描述
    sp.SignalType = signalTypeToString(tmpSignalType);
    sp.SignalPolarity = polarityToString(tmpSignalPolarity);
    sp.ArrayType = arrayTypeToString(tmpArrayType);
    sp.ArrayPolarity = polarityToString(tmpArrayPolarity);  // 注意：阵列极化与信号极化使用相同映射
    sp.CaliMethod = caliMethodToString(tmpCaliMethod);

    file.close();
    return sp;
}