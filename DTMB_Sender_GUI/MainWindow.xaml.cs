using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Windows.Storage.Pickers;

namespace DTMB_Sender_GUI
{
    public sealed partial class MainWindow : Window
    {
        [DllImport("winmm.dll")]
        private static extern uint timeBeginPeriod(uint uPeriod);

        [DllImport("winmm.dll")]
        private static extern uint timeEndPeriod(uint uPeriod);
        private SharedMemory? _shm;
        private ComplexFloat[]? _interleavedData;
        private int _numBatches;
        private int _batchSamples;
        private float _dataRate;
        private Thread? _senderThread;
        private volatile bool _isRunning;
        private volatile bool _isPaused;
        private volatile bool _shouldStop;
        private volatile bool _loopEnabled;
        private long _programStartTicks;
        private bool _programStarted;
        private Microsoft.UI.Dispatching.DispatcherQueue _dispatcherQueue;

        public MainWindow()
        {
            this.InitializeComponent();
            this.Title = "DTMB Signal Sender";

            // Save the dispatcher queue
            _dispatcherQueue = Microsoft.UI.Dispatching.DispatcherQueue.GetForCurrentThread();

            var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
            var windowId = Microsoft.UI.Win32Interop.GetWindowIdFromWindow(hwnd);
            var appWindow = Microsoft.UI.Windowing.AppWindow.GetFromWindowId(windowId);
            appWindow.Resize(new Windows.Graphics.SizeInt32(1050, 975));

            // Initialize loop state
            _loopEnabled = LoopCheckBox.IsChecked == true;

            AppendLog("========================================");
            AppendLog("  DTMB Signal Sender");
            AppendLog("========================================");
            AppendLog("[INFO] Select data file and click Start");
        }

        private bool IsSimMode => RadioSimMode.IsChecked == true;

        private void ModeRadioButtons_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            UpdateModeUI();
        }

        private void UpdateModeUI()
        {
            bool simMode = IsSimMode;
            SimFilePathBox.IsEnabled = simMode;
            BrowseSimBtn.IsEnabled = simMode;
            RealFilePathBox.IsEnabled = !simMode;
            BrowseRealBtn.IsEnabled = !simMode;
            RefChCombo.IsEnabled = !simMode;
            Sur1ChCombo.IsEnabled = !simMode;
            Sur2ChCombo.IsEnabled = !simMode;
        }

        private void LoopCheckBox_Checked(object sender, RoutedEventArgs e)
        {
            _loopEnabled = true;
        }

        private void LoopCheckBox_Unchecked(object sender, RoutedEventArgs e)
        {
            _loopEnabled = false;
        }

        private async void BrowseSimBtn_Click(object sender, RoutedEventArgs e)
        {
            var picker = new FileOpenPicker();
            WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(this));
            picker.SuggestedStartLocation = PickerLocationId.DocumentsLibrary;
            picker.FileTypeFilter.Add(".bin");
            picker.FileTypeFilter.Add("*");

            var file = await picker.PickSingleFileAsync();
            if (file != null)
            {
                SimFilePathBox.Text = file.Path;
            }
        }

        private async void BrowseRealBtn_Click(object sender, RoutedEventArgs e)
        {
            var picker = new FileOpenPicker();
            WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(this));
            picker.SuggestedStartLocation = PickerLocationId.DocumentsLibrary;
            picker.FileTypeFilter.Add(".bin");
            picker.FileTypeFilter.Add("*");

            var file = await picker.PickSingleFileAsync();
            if (file != null)
            {
                RealFilePathBox.Text = file.Path;
            }
        }

        private async void StartBtn_Click(object sender, RoutedEventArgs e)
        {
            if (!_isRunning)
            {
                await StartSending();
            }
            else if (!_isPaused)
            {
                PauseSending();
            }
            else
            {
                ResumeSending();
            }
        }

        private void StopBtn_Click(object sender, RoutedEventArgs e)
        {
            StopSending();
        }

        private async Task StartSending()
        {
            try
            {
                AppendLog("[INFO] Loading data...");

                // 释放之前的数据以避免内存累积
                if (_interleavedData != null)
                {
                    _interleavedData = null;
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                }

                float dataRate = ShmConstants.SAMPLE_RATE;
                float origFs = ShmConstants.SAMPLE_RATE; // 先给默认值
                if (IsSimMode)
                {
                    string path = SimFilePathBox.Text.Trim();
                    if (string.IsNullOrEmpty(path))
                    {
                        ShowError("Please select a simulation data file.");
                        return;
                    }
                    _interleavedData = await Task.Run(() =>
                        DataLoader.LoadSimulationData(path, out _numBatches, out _batchSamples));
                    dataRate = ShmConstants.SAMPLE_RATE;
                }
                else
                {
                    string path = RealFilePathBox.Text.Trim();
                    if (string.IsNullOrEmpty(path))
                    {
                        // 使用默认路径
                        path = "D:/CUDA_Program/testdata.bin";
                    }
                    int refCh = RefChCombo.SelectedIndex;
                    int sur1Ch = Sur1ChCombo.SelectedIndex;
                    int sur2Ch = Sur2ChCombo.SelectedIndex;
                    _interleavedData = await Task.Run(() =>
                        DataLoader.LoadMeasuredData(path, refCh, sur1Ch, sur2Ch,
                            out _numBatches, out _batchSamples, out origFs));
                    dataRate = origFs;
                }

                AppendLog($"[INFO] Data loaded: {_numBatches} batches, {_batchSamples} samples/batch");
                AppendLog($"[INFO] Data rate: {dataRate / 1e6:F2} MHz");
                _dataRate = dataRate;

                if (_shm == null)
                {
                    _shm = new SharedMemory();
                    if (!_shm.Create())
                    {
                        AppendLog("[ERROR] Failed to create shared memory");
                        _shm = null;
                        return;
                    }
                    AppendLog("[INFO] Shared memory created");
                }

                _shm.SetTotalBatches((uint)_numBatches);
                _shm.SetIsRunning(1);
                _shm.SetDataMode(IsSimMode ? 0u : 1u);

                _shouldStop = false;
                _isPaused = false;
                _isRunning = true;
                _programStarted = false;
                _loopEnabled = LoopCheckBox.IsChecked == true;

                StartBtn.Content = "Pause";
                StopBtn.IsEnabled = true;
                UpdateStatus("Sending...", InfoBarSeverity.Informational);

                _senderThread = new Thread(SenderThreadFunc)
                {
                    IsBackground = true
                };
                _senderThread.Start();
            }
            catch (Exception ex)
            {
                AppendLog($"[ERROR] {ex.Message}");
                ShowError($"Failed to start: {ex.Message}");
            }
        }

        private void PauseSending()
        {
            _isPaused = true;
            StartBtn.Content = "Resume";
            UpdateStatus("Paused", InfoBarSeverity.Warning);
            AppendLog("[INFO] Paused");
        }

        private void ResumeSending()
        {
            _isPaused = false;
            StartBtn.Content = "Pause";
            UpdateStatus("Sending...", InfoBarSeverity.Informational);
            AppendLog("[INFO] Resumed");
        }

        private void StopSending()
        {
            _shouldStop = true;
            _isPaused = false;
            _isRunning = false;

            if (_shm != null)
            {
                _shm.SetIsRunning(0);
            }

            // 释放数据以避免内存泄漏
            if (_interleavedData != null)
            {
                _interleavedData = null;
            }

            StartBtn.Content = "Start";
            StopBtn.IsEnabled = false;
            UpdateStatus("Idle", InfoBarSeverity.Informational);
            AppendLog("[INFO] Stopped");

            // 手动触发垃圾回收
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        private void PreciseSleepUntil(long targetTicks)
        {
            long nowTicks = Stopwatch.GetTimestamp();
            if (nowTicks >= targetTicks) return;

            double remainingSec = (targetTicks - nowTicks) / (double)Stopwatch.Frequency;
            if (remainingSec > 0.002)
            {
                Thread.Sleep((int)((remainingSec - 0.001) * 1000));
            }

            while (Stopwatch.GetTimestamp() < targetTicks)
            {
                Thread.Sleep(0);
            }
        }

        private void SenderThreadFunc()
        {
            timeBeginPeriod(1);

            int batchCount = 0;
            double batchDuration = _batchSamples / (double)_dataRate;
            long nextSendTicks = Stopwatch.GetTimestamp();
            bool firstBatch = true;

            while (!_shouldStop)
            {
                for (int batch = 0; batch < _numBatches && !_shouldStop; batch++)
                {
                    while (_isPaused && !_shouldStop)
                    {
                        Thread.Sleep(50);
                    }
                    if (_shouldStop) break;

                    if (!firstBatch)
                    {
                        PreciseSleepUntil(nextSendTicks);
                    }
                    firstBatch = false;

                    nextSendTicks = Stopwatch.GetTimestamp() + (long)(batchDuration * Stopwatch.Frequency);

                    if (!_shm!.LockTx())
                    {
                        continue;
                    }

                    if (_shm.GetIsRunning() == 0)
                    {
                        _shm.UnlockTx();
                        _shouldStop = true;
                        break;
                    }

                    if (!_programStarted)
                    {
                        _programStartTicks = Stopwatch.GetTimestamp();
                        _programStarted = true;
                    }

                    int startIdx = batch * _batchSamples * 3;
                    int count = _batchSamples * 3;
                    _shm.WriteComplexData(_interleavedData!, startIdx, count);
                    _shm.SetBatchIndex((uint)batch);

                    long elapsedUs = (long)((Stopwatch.GetTimestamp() - _programStartTicks) / (double)Stopwatch.Frequency * 1_000_000);
                    _shm.SetTimestamp((ulong)elapsedUs);

                    if (!_shm.UnlockTx())
                    {
                        continue;
                    }

                    batchCount++;

                    if (batchCount % 10 == 0 || batch == _numBatches - 1)
                    {
                        _dispatcherQueue.TryEnqueue(() =>
                        {
                            AppendLog($"[INFO] Sent batch {batch + 1}/{_numBatches} (Total {batchCount})");
                            Progressbar.Value = (double)(batch + 1) / _numBatches * 100;
                        });
                    }
                }

                if (_shouldStop) break;

                if (!_loopEnabled)
                {
                    _dispatcherQueue.TryEnqueue(() =>
                    {
                        _isRunning = false;
                        StartBtn.Content = "Start";
                        StopBtn.IsEnabled = false;
                        UpdateStatus("Completed", InfoBarSeverity.Success);
                        AppendLog("[INFO] Single play completed");
                        Progressbar.Value = 100;
                    });
                    break;
                }
                else
                {
                    firstBatch = true;
                    _dispatcherQueue.TryEnqueue(() =>
                    {
                        AppendLog("[INFO] Loop playback, restarting...");
                        Progressbar.Value = 0;
                    });
                }
            }

            timeEndPeriod(1);

            _dispatcherQueue.TryEnqueue(() =>
            {
                AppendLog("[INFO] Sender thread stopped");
            });
        }

        private void AppendLog(string message)
        {
            // Check if we're on the UI thread
            if (_dispatcherQueue.HasThreadAccess)
            {
                string timestamp = DateTime.Now.ToString("HH:mm:ss");
                LogTextBox.Text += $"[{timestamp}] {message}\n";
            }
            else
            {
                _dispatcherQueue.TryEnqueue(() => AppendLog(message));
            }
        }

        private void UpdateStatus(string message, InfoBarSeverity severity)
        {
            if (_dispatcherQueue.HasThreadAccess)
            {
                StatusInfoBar.Message = message;
                StatusInfoBar.Severity = severity;
            }
            else
            {
                _dispatcherQueue.TryEnqueue(() => UpdateStatus(message, severity));
            }
        }

        private async void ShowError(string message)
        {
            var dialog = new ContentDialog
            {
                Title = "Error",
                Content = message,
                CloseButtonText = "OK",
                XamlRoot = this.Content.XamlRoot
            };
            await dialog.ShowAsync();
        }
    }
}
