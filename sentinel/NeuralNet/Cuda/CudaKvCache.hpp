#ifndef CUDAKVCACHE_HPP
#define CUDAKVCACHE_HPP

#include "CudaMatmul.hpp"

/// <summary>
/// append only device KV cache for one attention layer
/// key and value are embeddingDim x maximumPositionCount with used prefix length
/// </summary>
class CudaKvCache {
public:
    CudaMatrix key;
    CudaMatrix value;
    int length;
    int maximumPositionCount;
    int embeddingDim;

    CudaKvCache();

    /// <summary>allocate capacity and reset length to zero</summary>
    void ensureCapacity(int embeddingDim, int maximumPositionCount);

    /// <summary>set length to zero keep capacity</summary>
    void reset();

    /// <summary>append packed key and value columns embedDim x stepCount</summary>
    void append(const CudaMatrix& keyStep, const CudaMatrix& valueStep);
};

#endif // CUDAKVCACHE_HPP
