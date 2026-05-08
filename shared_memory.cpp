#include "shared_memory.h"
#include <iostream>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>
#include <errno.h>
#endif

SharedMemory::SharedMemory() 
    : m_data(nullptr)
#ifdef _WIN32
    , m_hMapFile(nullptr)
    , m_hSemTxReady(nullptr)
    , m_hSemRxReady(nullptr)
#else
    , m_fd(-1)
    , m_semTxReady(nullptr)
    , m_semRxReady(nullptr)
#endif
{
}

SharedMemory::~SharedMemory() {
    close();
}

bool SharedMemory::create() {
#ifdef _WIN32
    m_hMapFile = CreateFileMappingW(
        INVALID_HANDLE_VALUE,
        nullptr,
        PAGE_READWRITE,
        0,
        sizeof(SharedMemoryData),
        SHARED_MEMORY_NAME
    );
    
    if (m_hMapFile == nullptr) {
        std::cerr << "[ERROR] CreateFileMapping failed: " << GetLastError() << std::endl;
        return false;
    }
    
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        std::cerr << "[WARNING] Shared memory already exists" << std::endl;
    }
    
    m_data = (SharedMemoryData*)MapViewOfFile(
        m_hMapFile,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        sizeof(SharedMemoryData)
    );
    
    if (m_data == nullptr) {
        std::cerr << "[ERROR] MapViewOfFile failed: " << GetLastError() << std::endl;
        CloseHandle(m_hMapFile);
        m_hMapFile = nullptr;
        return false;
    }
    
    m_hSemTxReady = CreateSemaphoreW(
        nullptr,
        0,
        1,
        SEMAPHORE_TX_READY_NAME
    );
    
    m_hSemRxReady = CreateSemaphoreW(
        nullptr,
        1,
        1,
        SEMAPHORE_RX_READY_NAME
    );
    
    if (m_hSemTxReady == nullptr || m_hSemRxReady == nullptr) {
        std::cerr << "[ERROR] CreateSemaphore failed" << std::endl;
        close();
        return false;
    }
    
    memset(m_data, 0, sizeof(SharedMemoryData));
    m_data->magic = MAGIC_NUMBER;
    m_data->is_running = 1;
    
#else
    shm_unlink(SHARED_MEMORY_NAME);
    sem_unlink(SEMAPHORE_TX_READY_NAME);
    sem_unlink(SEMAPHORE_RX_READY_NAME);
    
    m_fd = shm_open(SHARED_MEMORY_NAME, O_CREAT | O_RDWR, 0666);
    if (m_fd < 0) {
        std::cerr << "[ERROR] shm_open failed: " << strerror(errno) << std::endl;
        return false;
    }
    
    if (ftruncate(m_fd, sizeof(SharedMemoryData)) < 0) {
        std::cerr << "[ERROR] ftruncate failed: " << strerror(errno) << std::endl;
        close();
        return false;
    }
    
    m_data = (SharedMemoryData*)mmap(
        nullptr,
        sizeof(SharedMemoryData),
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        m_fd,
        0
    );
    
    if (m_data == MAP_FAILED) {
        std::cerr << "[ERROR] mmap failed: " << strerror(errno) << std::endl;
        close();
        return false;
    }
    
    m_semTxReady = sem_open(SEMAPHORE_TX_READY_NAME, O_CREAT, 0666, 0);
    m_semRxReady = sem_open(SEMAPHORE_RX_READY_NAME, O_CREAT, 0666, 1);
    
    if (m_semTxReady == SEM_FAILED || m_semRxReady == SEM_FAILED) {
        std::cerr << "[ERROR] sem_open failed" << std::endl;
        close();
        return false;
    }
    
    memset(m_data, 0, sizeof(SharedMemoryData));
    m_data->magic = MAGIC_NUMBER;
    m_data->is_running = 1;
#endif
    
    std::cout << "[INFO] Shared memory created successfully" << std::endl;
    return true;
}

bool SharedMemory::open() {
#ifdef _WIN32
    m_hMapFile = OpenFileMappingW(
        FILE_MAP_ALL_ACCESS,
        FALSE,
        SHARED_MEMORY_NAME
    );
    
    if (m_hMapFile == nullptr) {
        std::cerr << "[ERROR] OpenFileMapping failed: " << GetLastError() << std::endl;
        return false;
    }
    
    m_data = (SharedMemoryData*)MapViewOfFile(
        m_hMapFile,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        sizeof(SharedMemoryData)
    );
    
    if (m_data == nullptr) {
        std::cerr << "[ERROR] MapViewOfFile failed: " << GetLastError() << std::endl;
        CloseHandle(m_hMapFile);
        m_hMapFile = nullptr;
        return false;
    }
    
    m_hSemTxReady = OpenSemaphoreW(
        SYNCHRONIZE | SEMAPHORE_MODIFY_STATE,
        FALSE,
        SEMAPHORE_TX_READY_NAME
    );
    
    m_hSemRxReady = OpenSemaphoreW(
        SYNCHRONIZE | SEMAPHORE_MODIFY_STATE,
        FALSE,
        SEMAPHORE_RX_READY_NAME
    );
    
    if (m_hSemTxReady == nullptr || m_hSemRxReady == nullptr) {
        std::cerr << "[ERROR] OpenSemaphore failed" << std::endl;
        close();
        return false;
    }
    
    if (m_data->magic != MAGIC_NUMBER) {
        std::cerr << "[ERROR] Invalid magic number" << std::endl;
        close();
        return false;
    }
    
#else
    m_fd = shm_open(SHARED_MEMORY_NAME, O_RDWR, 0666);
    if (m_fd < 0) {
        std::cerr << "[ERROR] shm_open failed: " << strerror(errno) << std::endl;
        return false;
    }
    
    m_data = (SharedMemoryData*)mmap(
        nullptr,
        sizeof(SharedMemoryData),
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        m_fd,
        0
    );
    
    if (m_data == MAP_FAILED) {
        std::cerr << "[ERROR] mmap failed: " << strerror(errno) << std::endl;
        close();
        return false;
    }
    
    m_semTxReady = sem_open(SEMAPHORE_TX_READY_NAME, 0);
    m_semRxReady = sem_open(SEMAPHORE_RX_READY_NAME, 0);
    
    if (m_semTxReady == SEM_FAILED || m_semRxReady == SEM_FAILED) {
        std::cerr << "[ERROR] sem_open failed" << std::endl;
        close();
        return false;
    }
    
    if (m_data->magic != MAGIC_NUMBER) {
        std::cerr << "[ERROR] Invalid magic number" << std::endl;
        close();
        return false;
    }
#endif
    
    std::cout << "[INFO] Shared memory opened successfully" << std::endl;
    return true;
}

void SharedMemory::close() {
#ifdef _WIN32
    if (m_data) {
        UnmapViewOfFile(m_data);
        m_data = nullptr;
    }
    if (m_hMapFile) {
        CloseHandle(m_hMapFile);
        m_hMapFile = nullptr;
    }
    if (m_hSemTxReady) {
        CloseHandle(m_hSemTxReady);
        m_hSemTxReady = nullptr;
    }
    if (m_hSemRxReady) {
        CloseHandle(m_hSemRxReady);
        m_hSemRxReady = nullptr;
    }
#else
    if (m_data && m_data != MAP_FAILED) {
        munmap(m_data, sizeof(SharedMemoryData));
        m_data = nullptr;
    }
    if (m_fd >= 0) {
        ::close(m_fd);
        m_fd = -1;
    }
    if (m_semTxReady && m_semTxReady != SEM_FAILED) {
        sem_close(m_semTxReady);
        m_semTxReady = nullptr;
    }
    if (m_semRxReady && m_semRxReady != SEM_FAILED) {
        sem_close(m_semRxReady);
        m_semRxReady = nullptr;
    }
#endif
}

bool SharedMemory::lock_tx() {
#ifdef _WIN32
    DWORD result = WaitForSingleObject(m_hSemRxReady, INFINITE);
    return result == WAIT_OBJECT_0;
#else
    return sem_wait(m_semRxReady) == 0;
#endif
}

bool SharedMemory::unlock_tx() {
#ifdef _WIN32
    return ReleaseSemaphore(m_hSemTxReady, 1, nullptr) != FALSE;
#else
    return sem_post(m_semTxReady) == 0;
#endif
}

bool SharedMemory::lock_rx() {
#ifdef _WIN32
    DWORD result = WaitForSingleObject(m_hSemTxReady, INFINITE);
    return result == WAIT_OBJECT_0;
#else
    return sem_wait(m_semTxReady) == 0;
#endif
}

bool SharedMemory::unlock_rx() {
#ifdef _WIN32
    return ReleaseSemaphore(m_hSemRxReady, 1, nullptr) != FALSE;
#else
    return sem_post(m_semRxReady) == 0;
#endif
}
