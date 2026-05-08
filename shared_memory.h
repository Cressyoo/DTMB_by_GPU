#ifndef SHARED_MEMORY_H
#define SHARED_MEMORY_H

#include <cstdint>
#include <complex>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>
#endif

// Signal parameters
#define SAMPLE_RATE 7560000.0f
#define FRAME_LEN 4200
#define BATCH_SIZE 756000
#define MAX_INPUT_BATCH_SIZE 1000000
#define NUM_CHANNELS 3
#define DATA_ARRAY_SIZE (MAX_INPUT_BATCH_SIZE * NUM_CHANNELS)

// Shared memory names
#ifdef _WIN32
#define SHARED_MEMORY_NAME L"DTMB_SharedMemory"
#define SEMAPHORE_TX_READY_NAME L"DTMB_SemTxReady"
#define SEMAPHORE_RX_READY_NAME L"DTMB_SemRxReady"
#else
#define SHARED_MEMORY_NAME "/dtmb_shared_memory"
#define SEMAPHORE_TX_READY_NAME "/dtmb_sem_tx_ready"
#define SEMAPHORE_RX_READY_NAME "/dtmb_sem_rx_ready"
#endif

struct SharedMemoryData {
    uint32_t magic;
    uint32_t batch_index;
    uint32_t total_batches;
    uint32_t is_running;
    uint64_t timestamp_us;
    uint32_t data_mode;  // 0: 3-channel simulation, 1: 8-channel measured
    
    std::complex<float> data[DATA_ARRAY_SIZE];
};

#define MAGIC_NUMBER 0x44544D42

class SharedMemory {
public:
    SharedMemory();
    ~SharedMemory();
    
    bool create();
    bool open();
    void close();
    
    SharedMemoryData* data() { return m_data; }
    
    bool lock_tx();
    bool unlock_tx();
    
    bool lock_rx();
    bool unlock_rx();
    
private:
    SharedMemoryData* m_data;
    
#ifdef _WIN32
    HANDLE m_hMapFile;
    HANDLE m_hSemTxReady;
    HANDLE m_hSemRxReady;
#else
    int m_fd;
    sem_t* m_semTxReady;
    sem_t* m_semRxReady;
#endif
};

#endif
