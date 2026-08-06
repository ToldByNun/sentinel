#ifndef ROTARYEMBEDDING_HPP
#define ROTARYEMBEDDING_HPP

#include "../Math/Matrix.hpp"

#include <vector>

/// <summary>
/// rotary position embedding for multi head attention
/// rotates feature pairs in each head by position dependent angles
/// </summary>
class RotaryEmbedding {
public:
    static constexpr float DefaultBase = 10000.0f;

    int headDimension;
    int maximumPositionCount;
    int pairCount;
    float base = 0.0f;
    std::vector<float> cosTable;
    std::vector<float> sinTable;

    /// <summary>build cos/sin tables for positions 0 .. maximumPositionCount-1 (HF rope_theta → base)</summary>
    RotaryEmbedding(int headDimension, int maximumPositionCount, float base = DefaultBase);

    /// <summary>default empty rope</summary>
    RotaryEmbedding();

    /// <summary>rotate Q or K in place tensor shape embeddingDim x sequenceLength</summary>
    void rotateInPlace(Matrix& tensor, int headCount) const;

    /// <summary>inverse rotation for gradients in place</summary>
    void rotateInverseInPlace(Matrix& tensor, int headCount) const;

private:
    /// <summary>cos/sin lookup for one position and pair index</summary>
    float cosAt(size_t position, size_t pairIndex) const;

    /// <summary>sin lookup for one position and pair index</summary>
    float sinAt(size_t position, size_t pairIndex) const;
};

#endif // ROTARYEMBEDDING_HPP
