#ifndef MEANPOOL_HPP
#define MEANPOOL_HPP

#include "../Math/Matrix.hpp"

#include <vector>

class MeanPool {
public:
	static Matrix meanPool(const Matrix& embeddings);
};

#endif // MEANPOOL_HPP