using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace DTMB_Sender_GUI
{
    public class DataLoader
    {
        public static ComplexFloat[] LoadSimulationData(string filePath, out int numBatches, out int batchSamples)
        {
            using FileStream fs = new(filePath, FileMode.Open, FileAccess.Read);
            using BinaryReader br = new(fs);
            
            long fileLength = fs.Length;
            int totalSamples = (int)(fileLength / (3 * 8));
            batchSamples = ShmConstants.BATCH_SIZE;
            numBatches = totalSamples / batchSamples;

            ComplexFloat[] interleaved = new ComplexFloat[totalSamples * 3];
            for (int i = 0; i < totalSamples; i++)
            {
                float ch0_re = br.ReadSingle();
                float ch0_im = br.ReadSingle();
                float ch1_re = br.ReadSingle();
                float ch1_im = br.ReadSingle();
                float ch2_re = br.ReadSingle();
                float ch2_im = br.ReadSingle();

                interleaved[i * 3] = new ComplexFloat(ch0_re, ch0_im);
                interleaved[i * 3 + 1] = new ComplexFloat(ch1_re, ch1_im);
                interleaved[i * 3 + 2] = new ComplexFloat(ch2_re, ch2_im);
            }

            return interleaved;
        }

        public static ComplexFloat[] LoadMeasuredData(string filePath, int refCh, int sur1Ch, int sur2Ch,
            out int numBatches, out int batchSamples, out float originalFs)
        {
            SysParm fileHead = ReadFileHeader(filePath);
            originalFs = fileHead.fs;

            // 直接使用 MAX_INPUT_BATCH_SIZE (1,000,000) 作为每批的样本数
            int samplesPerBatch = ShmConstants.MAX_INPUT_BATCH_SIZE;

            numBatches = (int)(fileHead.DatLen / ((ulong)(fileHead.ChanNum * 2)) / (ulong)samplesPerBatch);
            if (numBatches <= 0) numBatches = 1;

            int totalReadSamples = numBatches * samplesPerBatch;
            batchSamples = samplesPerBatch;

            int nchan = fileHead.ChanNum;
            int[] channels = { refCh, sur1Ch, sur2Ch };

            ComplexFloat[] interleaved = new ComplexFloat[totalReadSamples * 3];

            using FileStream fs = new(filePath, FileMode.Open, FileAccess.Read);
            fs.Seek(fileHead.FileHeadLen, SeekOrigin.Begin);

            int batchSizeInt16 = samplesPerBatch * 2 * nchan;
            short[] buffer = new short[batchSizeInt16];
            byte[] byteBuffer = new byte[batchSizeInt16 * 2];

            for (int batch = 0; batch < numBatches; batch++)
            {
                int bytesRead = fs.Read(byteBuffer, 0, byteBuffer.Length);
                if (bytesRead < byteBuffer.Length)
                {
                    numBatches = batch;
                    break;
                }
                Buffer.BlockCopy(byteBuffer, 0, buffer, 0, bytesRead);

                for (int i = 0; i < samplesPerBatch; i++)
                {
                    for (int ch_idx = 0; ch_idx < 3; ch_idx++)
                    {
                        int ch = channels[ch_idx];
                        int buffIdx = i * (2 * nchan) + 2 * ch;
                        float real_val = buffer[buffIdx];
                        float imag_val = buffer[buffIdx + 1];
                        interleaved[batch * samplesPerBatch * 3 + i * 3 + ch_idx] = new ComplexFloat(real_val, imag_val);
                    }
                }
            }

            return interleaved;
        }

        private static SysParm ReadFileHeader(string filePath)
        {
            SysParm sp = new();
            using FileStream fs = new(filePath, FileMode.Open, FileAccess.Read);
            using BinaryReader br = new(fs);

            sp.VersionNo = br.ReadInt32();
            sp.FileHeadLen = br.ReadInt32();

            int tmpSignalType = br.ReadInt32();
            int tmpSignalPolarity = br.ReadInt32();

            float f0_mhz = br.ReadSingle();
            sp.f0 = f0_mhz * 1e6;

            byte[] signalModeBuf = br.ReadBytes(16);
            sp.SignalMode = Encoding.ASCII.GetString(signalModeBuf);

            byte[] rxTxIDBuf = br.ReadBytes(32);
            sp.RxTxID = Encoding.ASCII.GetString(rxTxIDBuf);

            double[] rxPos = new double[3];
            double[] txPos = new double[3];
            for (int i = 0; i < 3; i++) rxPos[i] = br.ReadDouble();
            for (int i = 0; i < 3; i++) txPos[i] = br.ReadDouble();

            int tmpArrayType = br.ReadInt32();
            sp.ThetaNormal = br.ReadSingle();
            int tmpArrayPolarity = br.ReadInt32();
            sp.ArrayDimension = br.ReadInt32();

            int antCordSize = sp.ArrayDimension * 3;
            float[] antCord = new float[antCordSize];
            for (int i = 0; i < antCordSize; i++) antCord[i] = br.ReadSingle();

            int tmpCaliMethod = br.ReadInt32();
            sp.CaliStepAngle = br.ReadSingle();

            int numAngles = (int)Math.Round(360.0f / sp.CaliStepAngle);
            int totalFloat = sp.ArrayDimension * numAngles * 2;
            float[] caliRaw = new float[totalFloat];
            for (int i = 0; i < totalFloat; i++) caliRaw[i] = br.ReadSingle();

            sp.fs = br.ReadSingle();
            sp.StartSampleTime = br.ReadDouble();
            sp.DatLen = br.ReadUInt64();
            sp.ChanNum = br.ReadInt32();

            int attSize = 2 * sp.ChanNum;
            float[] att = new float[attSize];
            for (int i = 0; i < attSize; i++) att[i] = br.ReadSingle();

            return sp;
        }
    }

    public struct SysParm
    {
        public int VersionNo;
        public int FileHeadLen;
        public string SignalMode;
        public string RxTxID;
        public double f0;
        public float ThetaNormal;
        public int ArrayDimension;
        public float CaliStepAngle;
        public float fs;
        public double StartSampleTime;
        public ulong DatLen;
        public int ChanNum;
    }
}
