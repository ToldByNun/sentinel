#ifndef CUDAMATMUL_HPP
#define CUDAMATMUL_HPP

#include "../Math/Matrix.hpp"

#include <cstddef>

/// <summary>reusable device float buffer grows only when capacity is too small</summary>
class CudaDeviceBuffer {
public:
    float* deviceData;
    size_t capacityBytes;

    CudaDeviceBuffer();
    ~CudaDeviceBuffer();

    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

    /// <summary>allocate when capacityBytes is below requiredBytes</summary>
    void ensureCapacity(size_t requiredBytes);

    /// <summary>host to device copy into already capacious buffer</summary>
    void copyFromHost(const float* hostData, size_t byteCount);

    /// <summary>device to host copy from already capacious buffer</summary>
    void copyToHost(float* hostData, size_t byteCount) const;

    /// <summary>release device memory</summary>
    void free();
};

/// <summary>split timings for one host driven multiply</summary>
class CudaMatmulTiming {
public:
    double ensureCapacityMilliseconds;
    double hostToDeviceMilliseconds;
    double kernelMilliseconds;
    double deviceToHostMilliseconds;

    double totalMilliseconds() const;
};

/// <summary>
/// CUDA row major GEMM C = A * B using shared memory tiling
/// keeps device buffers across calls so malloc free is not the hot path
/// </summary>
class CudaMatmul {
public:
    CudaDeviceBuffer deviceLeft;
    CudaDeviceBuffer deviceRight;
    CudaDeviceBuffer deviceOut;

    CudaMatmul();

    /// <summary>true if a CUDA device is available</summary>
    static bool isAvailable();

    /// <summary>C = A * B on GPU reusing owned device buffers</summary>
    void multiplyInto(const Matrix& left, const Matrix& right, Matrix& out);

    /// <summary>C = A * B on GPU with split stage timings</summary>
    void multiplyIntoTimed(const Matrix& left, const Matrix& right, Matrix& out, CudaMatmulTiming& timing);

    /// <summary>C = A * B on GPU</summary>
    Matrix multiply(const Matrix& left, const Matrix& right);

    /// <summary>thread local workspace for static callers</summary>
    static CudaMatmul& sharedWorkspace();

    /// <summary>C = A * B via sharedWorkspace</summary>
    static void multiplyIntoShared(const Matrix& left, const Matrix& right, Matrix& out);

    /// <summary>compare CPU gemm vs CUDA with cold versus warm and stage timings</summary>
    static void runSmokeDemo(size_t matrixSize = 512);

private:
    friend class CudaDeviceBuffer;

    static constexpr int tileSize = 16;

    /// <summary>throw runtime_error when a CUDA call failed</summary>
    static void throwIfCudaFailed(int status, const char* operationName);

    /// <summary>fill matrix with deterministic values in minus one to one</summary>
    static Matrix makeRandomMatrix(size_t rowCount, size_t columnCount, unsigned seed);

    /// <summary>maximum absolute difference between same shape matrices</summary>
    static float maximumAbsoluteDifference(const Matrix& left, const Matrix& right);

    /// <summary>grow owned buffers then copy launch and copy back optionally filling timing</summary>
    void multiplyIntoInternal(const Matrix& left, const Matrix& right, Matrix& out, CudaMatmulTiming* timing);

#ifdef __CUDACC__
    /// <summary>device body for shared memory tiled matmul CUDA forbids __global__ members</summary>
    __device__ static void runSharedMemoryMatmul(const float* left, const float* right, float* out, int rowCount, int columnCount, int sharedCount);

    friend __global__ void CudaMatmulSharedMemoryEntry(const float* left, const float* right, float* out, int rowCount, int columnCount, int sharedCount);
#endif
};

#ifdef __CUDACC__
/// <summary>CUDA language requires a free __global__ entry trampoline into CudaMatmul</summary>
__global__ void CudaMatmulSharedMemoryEntry(const float* left, const float* right, float* out, int rowCount, int columnCount, int sharedCount);
#endif

#endif // CUDAMATMUL_HPP
