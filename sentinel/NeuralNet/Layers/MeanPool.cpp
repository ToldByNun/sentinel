#include "MeanPool.hpp"

Matrix MeanPool::meanPool(const Matrix& embeddings) {
	const size_t dim = embeddings.data.size();
	const size_t seqLen = embeddings.data[0].size();

	printf("dim, seqLen: %i %i\n", dim, seqLen);

	Matrix pooled;
	pooled.data = std::vector<std::vector<float>>(dim, std::vector<float>(1.0f, 0.0f));

	printf("data size: %i\n", pooled.data.size());

	for (size_t d = 0; d < dim; ++d) {
		float sum = 0.0f;
		for (size_t t = 0; t < seqLen; ++t) {
			sum += embeddings.data[d][t];
		}
		pooled.data[d][0] = sum / static_cast<float>(seqLen);
	}
	return pooled;
}