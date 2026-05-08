// ReadSrcFileHead.h
#ifndef READ_SRC_FILE_HEAD_H
#define READ_SRC_FILE_HEAD_H

#include <string>
#include <vector>
#include <complex>
#include <cstdint>

// 结构体定义，对应MATLAB的SysParm
struct SysParm {
    int32_t VersionNo;                          // 帧头版本号
    int32_t FileHeadLen;                         // 帧头总长度(字节)
    std::string SignalType;                       // 信号类型字符串描述
    std::string SignalPolarity;                   // 信号极化字符串描述
    double f0;                                    // 信号中心频率 (Hz)
    std::string SignalMode;                       // 信号帧头模式 (16字符)
    std::string RxTxID;                           // 收发对编号 (32字符)
    std::vector<double> RxPos;                    // 接收站位置信息 [x,y,z]
    std::vector<double> TxPos;                    // 发射站位置信息 [x,y,z]
    std::string ArrayType;                         // 阵列形式字符串描述
    float ThetaNormal;                             // 阵列法向(度)
    std::string ArrayPolarity;                     // 阵列极化字符串描述
    int32_t ArrayDimension;                         // 阵元维数
    std::vector<float> AntCord;                     // 各个阵元的坐标，长度为 ArrayDimension*3，行主序
    std::string CaliMethod;                         // 校准方法字符串描述
    float CaliStepAngle;                            // 插值法角度间隔(度)
    std::vector<std::complex<float>> CaliMatrix;    // 校准矩阵，长度为 ArrayDimension * numAngles，行主序
    float fs;                                        // 系统采样率 (Hz)
    double StartSampleTime;                          // 采样起始时间UNIX
    uint64_t DatLen;                                 // 文件数据体的长度
    int32_t ChanNum;                                 // 接收机通道个数
    std::vector<float> Att;                          // 接收机衰减，长度为 2*ChanNum，列主序
};

// 枚举定义（可选，用于参数传递和内部使用）
enum class SignalTypeEnum : int32_t {
    UNKNOWN = -1,
    DTMB = 0,
    CMMB = 1,
    FM = 2
};

enum class PolarityEnum : int32_t {
    UNKNOWN = -1,
    VERTICAL = 0,    // V
    HORIZONTAL = 1   // H
};

enum class ArrayTypeEnum : int32_t {
    CIRCULAR = 0,    // 圆阵
    LINEAR = 1       // 线阵
};

enum class CaliMethodEnum : int32_t {
    NO_CALIBRATION = -1,
    DECOMPOSITION = 0,  // 解算法
    INTERPOLATION = 1,  // 插值法
    BOTH = 2            // 解算法和插值法
};

// 辅助函数声明：将枚举转换为字符串
std::string signalTypeToString(int32_t type);
std::string polarityToString(int32_t pol);
std::string arrayTypeToString(int32_t type);
std::string caliMethodToString(int32_t method);

// 重载版本，接受枚举类型
inline std::string signalTypeToString(SignalTypeEnum type) {
    return signalTypeToString(static_cast<int32_t>(type));
}
inline std::string polarityToString(PolarityEnum pol) {
    return polarityToString(static_cast<int32_t>(pol));
}
inline std::string arrayTypeToString(ArrayTypeEnum type) {
    return arrayTypeToString(static_cast<int32_t>(type));
}
inline std::string caliMethodToString(CaliMethodEnum method) {
    return caliMethodToString(static_cast<int32_t>(method));
}

// 主函数声明：读取文件头并填充SysParm结构体
// 输入参数：
//   FilePath: 原始文件路径
// 返回值：
//   SysParm结构体，包含所有解析出的参数
// 异常：
//   当文件无法打开或读取失败时抛出 std::runtime_error
SysParm func_ReadSrcFileHead(const std::string& FilePath);

// 版本历史注释（保留在头文件中便于查阅）
/*
 * 版本历史:
 *   20191016: 程序创建
 *   20200213: 将输出参数修改为结构体输出，便于参数的保存
 *   202403XX: 转换为C++实现
 */

#endif // READ_SRC_FILE_HEAD_H