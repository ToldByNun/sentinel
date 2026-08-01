#ifndef CUDALANGUAGEMODEL_HPP
#define CUDALANGUAGEMODEL_HPP

#include "../Network/LanguageModel.hpp"
#include "CudaKvCache.hpp"
#include "CudaMatmul.hpp"
#include "CudaRMSNorm.hpp"
#include "CudaTransformerBlock.hpp"

#include <vector>

/// <summary>
/// device resident causal LM forward
/// token embed -> CudaTransformerBlock x N -> final RMSNorm -> vocab projection
/// </summary>
class CudaLanguageModel {
public:
    CudaMatrix tokenEmbeddingWeight;
    std::vector<CudaTransformerBlock> blocks;
    CudaRMSNorm finalNorm;
    CudaMatrix projectionWeight;
    CudaMatrix projectionBias;

    CudaMatrix hidden;
    CudaMatrix normalized;
    CudaMatrix logits;
    CudaIntBuffer tokenIdsBuffer;
    std::vector<CudaKvCache> kvCaches;
    int maximumPositionCount;

    CudaLanguageModel();

    /// <summary>upload all host LM parameters once</summary>
    void uploadFrom(const LanguageModel& host);

    /// <summary>create device LM from host LM</summary>
    static CudaLanguageModel createFrom(const LanguageModel& host);

    /// <summary>logits vocabSize x sequenceLength staying on device</summary>
    void forwardInto(const std::vector<int>& tokenIds, CudaMatrix& outLogits);

    /// <summary>logits downloaded to host</summary>
    Matrix forward(const std::vector<int>& tokenIds);

    /// <summary>reset and allocate one KV cache per block</summary>
    void resetKvCaches();

    /// <summary>prefill prefix into KV caches writing full-prefix logits</summary>
    void prefillInto(const std::vector<int>& tokenIds, CudaMatrix& outLogits);

    /// <summary>decode one new token against KV caches writing vocab x 1 logits</summary>
    void decodeInto(int tokenId, CudaMatrix& outLogits);

    /// <summary>compare CPU LanguageModel forward vs device forward</summary>
    static void runSmokeDemo(int vocabularySize = 128, int embeddingDim = 64, int sequenceLength = 32, int blockCount = 2, int headCount = 4);

    /// <summary>compare prefill plus decode last token vs full forward last column</summary>
    static void runKvCacheSmokeDemo(int vocabularySize = 128, int embeddingDim = 64, int sequenceLength = 32, int blockCount = 2, int headCount = 4);
};

#endif // CUDALANGUAGEMODEL_HPP
