#ifndef CUDALANGUAGEMODEL_HPP
#define CUDALANGUAGEMODEL_HPP

#include "../Network/LanguageModel.hpp"
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

    /// <summary>compare CPU LanguageModel forward vs device forward</summary>
    static void runSmokeDemo(int vocabularySize = 128, int embeddingDim = 64, int sequenceLength = 32, int blockCount = 2, int headCount = 4);
};

#endif // CUDALANGUAGEMODEL_HPP
