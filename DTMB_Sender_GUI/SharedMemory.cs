using System;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Threading;

namespace DTMB_Sender_GUI
{
    public static class ShmConstants
    {
        public const float SAMPLE_RATE = 7560000.0f;
        public const int FRAME_LEN = 4200;
        public const int BATCH_SIZE = 756000;
        public const int MAX_INPUT_BATCH_SIZE = 1000000;
        public const int NUM_CHANNELS = 3;
        public const int DATA_ARRAY_SIZE = MAX_INPUT_BATCH_SIZE * NUM_CHANNELS;
        public const uint MAGIC_NUMBER = 0x44544D42;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct SharedMemoryHeader
    {
        public uint magic;
        public uint batch_index;
        public uint total_batches;
        public uint is_running;
        public ulong timestamp_us;
        public uint data_mode;  // 0: 3-channel simulation, 1: 8-channel measured
    }

    public class SharedMemory : IDisposable
    {
        private const string ShmName = "DTMB_SharedMemory";
        private const string SemTxReadyName = "DTMB_SemTxReady";
        private const string SemRxReadyName = "DTMB_SemRxReady";

        private MemoryMappedFile? _mmf;
        private MemoryMappedViewAccessor? _accessor;
        private Semaphore? _semTxReady;
        private Semaphore? _semRxReady;

        private const int HeaderSize = 28;  // 新增data_mode字段，增加4字节
        private const int DataSize = ShmConstants.DATA_ARRAY_SIZE * 8;

        public bool Create()
        {
            try
            {
                _mmf = MemoryMappedFile.CreateOrOpen(ShmName, HeaderSize + DataSize, MemoryMappedFileAccess.ReadWrite);
                _accessor = _mmf.CreateViewAccessor();

                _semTxReady = new Semaphore(0, 1, SemTxReadyName, out bool txCreated);
                _semRxReady = new Semaphore(1, 1, SemRxReadyName, out bool rxCreated);

                _accessor.Write(0, ShmConstants.MAGIC_NUMBER);
                _accessor.Write(4, (uint)0);
                _accessor.Write(8, (uint)0);
                _accessor.Write(12, (uint)1);
                _accessor.Write(16, (ulong)0);

                return true;
            }
            catch
            {
                return false;
            }
        }

        public bool LockTx()
        {
            try
            {
                // Wait up to 5 seconds, avoid infinite waiting
                return _semRxReady!.WaitOne(5000);
            }
            catch
            {
                return false;
            }
        }

        public bool UnlockTx()
        {
            try
            {
                _semTxReady!.Release(1);
                return true;
            }
            catch
            {
                return false;
            }
        }

        public void SetBatchIndex(uint index)
        {
            _accessor!.Write(4, index);
        }

        public void SetTotalBatches(uint total)
        {
            _accessor!.Write(8, total);
        }

        public void SetIsRunning(uint running)
        {
            _accessor!.Write(12, running);
        }

        public void SetTimestamp(ulong us)
        {
            _accessor!.Write(16, us);
        }

        public uint GetIsRunning()
        {
            return _accessor!.ReadUInt32(12);
        }

        public void SetDataMode(uint mode)
        {
            _accessor!.Write(24, mode);
        }

        public uint GetDataMode()
        {
            return _accessor!.ReadUInt32(24);
        }

        public void WriteComplexData(ComplexFloat[] data, int srcOffset, int count)
        {
            _accessor!.WriteArray(HeaderSize, data, srcOffset, count);
        }

        public void WriteComplexData(ComplexFloat[] data)
        {
            WriteComplexData(data, 0, data.Length);
        }

        public void Dispose()
        {
            _accessor?.Dispose();
            _mmf?.Dispose();
            _semTxReady?.Dispose();
            _semRxReady?.Dispose();
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ComplexFloat
    {
        public float Real;
        public float Imag;

        public ComplexFloat(float real, float imag)
        {
            Real = real;
            Imag = imag;
        }

        public static implicit operator ComplexFloat((float re, float im) v) => new(v.re, v.im);
    }
}
