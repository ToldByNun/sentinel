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
    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept;
    CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept;

    /// <summary>allocate when capacityBytes is below requiredBytes</summary>
    void ensureCapacity(size_t requiredBytes);

    /// <summary>host to device copy into already capacious buffer</summary>
    void copyFromHost(const float* hostData, size_t byteCount);

    /// <summary>host to device raw byte copy into already capacious buffer</summary>
    void copyBytesFromHost(const void* hostData, size_t byteCount);

    /// <summary>device to host copy from already capacious buffer</summary>
    void copyToHost(float* hostData, size_t byteCount) const;

    /// <summary>release device memory</summary>
    void free();
};

/// <summary>reusable device int buffer grows only when capacity is too small</summary>
class CudaIntBuffer {
public:
    int* deviceData;
    size_t capacityCount;

    CudaIntBuffer();
    ~CudaIntBuffer();

    CudaIntBuffer(const CudaIntBuffer&) = delete;
    CudaIntBuffer& operator=(const CudaIntBuffer&) = delete;
    CudaIntBuffer(CudaIntBuffer&& other) noexcept;
    CudaIntBuffer& operator=(CudaIntBuffer&& other) noexcept;

    /// <summary>allocate when capacityCount is below requiredCount</summary>
    void ensureCapacity(size_t requiredCount);

    /// <summary>host to device copy into already capacious buffer</summary>
    void copyFromHost(const int* hostData, size_t count);

    /// <summary>release device memory</summary>
    void free();
};

/// <summary>row major float matrix that lives on the device</summary>
class CudaMatrix {
public:
    size_t rows;
    size_t cols;
    CudaDeviceBuffer buffer;

    CudaMatrix();
    CudaMatrix(const CudaMatrix&) = delete;
    CudaMatrix& operator=(const CudaMatrix&) = delete;
    CudaMatrix(CudaMatrix&& other) noexcept;
    CudaMatrix& operator=(CudaMatrix&& other) noexcept;

    /// <summary>true if rows or cols is zero</summary>
    bool empty() const;

    /// <summary>rows * cols</summary>
    size_t elementCount() const;

    /// <summary>elementCount * sizeof float</summary>
    size_t byteCount() const;

    /// <summary>set shape and ensure device capacity</summary>
    void ensureSize(size_t rowCount, size_t columnCount);

    /// <summary>release device memory and clear shape</summary>
    void free();

    /// <summary>copy host matrix to device growing capacity if needed</summary>
    void upload(const Matrix& host);

    /// <summary>synchronize then copy device matrix to host</summary>
    void downloadInto(Matrix& host) const;

    /// <summary>synchronize then return host copy</summary>
    Matrix download() const;

    /// <summary>C = op(A) * op(B) entirely on device no host copies</summary>
    static void multiplyInto(const CudaMatrix& left, const CudaMatrix& right, CudaMatrix& out, bool transposeLeft = false, bool transposeRight = false);

    /// <summary>C = op(A) * op(B) on device measuring kernel time with cuda events</summary>
    static void multiplyIntoTimed(const CudaMatrix& left, const CudaMatrix& right, CudaMatrix& out, double& kernelMilliseconds, bool transposeLeft = false, bool transposeRight = false);
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
/// CUDA row major GEMM C = A * B preferring cuBLASLt TF32 with shared memory fallback
/// host path reuses scratch buffers device path uses CudaMatrix
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

    /// <summary>compare CPU gemm vs CUDA host path and device resident path</summary>
    static void runSmokeDemo(size_t matrixSize = 512);

private:
    friend class CudaDeviceBuffer;
    friend class CudaIntBuffer;
    friend class CudaMatrix;
    friend class CudaOps;
    friend class CudaFeedForward;
    friend class CudaRMSNorm;
    friend class CudaCausalSelfAttention;
    friend class CudaFlashAttention;
    friend class CudaTransformerBlock;
    friend class CudaLanguageModel;
    friend class CudaAdam;

    static constexpr int tileSize = 16;

    /// <summary>throw runtime_error when a CUDA call failed</summary>
    static void throwIfCudaFailed(int status, const char* operationName);

    /// <summary>throw runtime_error when a cuBLASLt call failed</summary>
    static void throwIfCublasLtFailed(int status, const char* operationName);

    /// <summary>fill matrix with deterministic values in minus one to one</summary>
    static Matrix makeRandomMatrix(size_t rowCount, size_t columnCount, unsigned seed);

    /// <summary>maximum absolute difference between same shape matrices</summary>
    static float maximumAbsoluteDifference(const Matrix& left, const Matrix& right);

    /// <summary>launch cuBLASLt TF32 row major matmul returning false on any failure</summary>
    static bool launchCublasLtMatmul(const float* deviceLeft, const float* deviceRight, float* deviceOut, int rowCount, int columnCount, int sharedCount, bool transposeLeft, bool transposeRight, double* kernelMilliseconds);

    /// <summary>launch shared memory matmul kernel optionally filling kernelMilliseconds</summary>
    static void launchSharedMemoryMatmulKernel(const float* deviceLeft, const float* deviceRight, float* deviceOut, int rowCount, int columnCount, int sharedCount, bool transposeLeft, bool transposeRight, double* kernelMilliseconds);

    /// <summary>launch GEMM preferring cuBLASLt TF32 falling back to shared memory kernel</summary>
    static void launchSharedMemoryMatmul(const float* deviceLeft, const float* deviceRight, float* deviceOut, int rowCount, int columnCount, int sharedCount, bool transposeLeft, bool transposeRight, double* kernelMilliseconds);

    /// <summary>grow owned buffers then copy launch and copy back optionally filling timing</summary>
    void multiplyIntoInternal(const Matrix& left, const Matrix& right, Matrix& out, CudaMatmulTiming* timing);

#ifdef __CUDACC__
    /// <summary>device body for shared memory tiled matmul CUDA forbids __global__ members</summary>
    __device__ static void runSharedMemoryMatmul(const float* left, const float* right, float* out, int rowCount, int columnCount, int sharedCount, bool transposeLeft, bool transposeRight);

    friend __global__ void CudaMatmulSharedMemoryEntry(const float* left, const float* right, float* out, int rowCount, int columnCount, int sharedCount, bool transposeLeft, bool transposeRight);
#endif
};

#ifdef __CUDACC__
/// <summary>CUDA language requires a free __global__ entry trampoline into CudaMatmul</summary>
__global__ void CudaMatmulSharedMemoryEntry(const float* left, const float* right, float* out, int rowCount, int columnCount, int sharedCount, bool transposeLeft, bool transposeRight);
#endif

#endif // CUDAMATMUL_HPP
