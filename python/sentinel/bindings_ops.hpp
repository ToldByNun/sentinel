#ifndef SENTINEL_PYTHON_BINDINGS_OPS_HPP
#define SENTINEL_PYTHON_BINDINGS_OPS_HPP

#include <nanobind/nanobind.h>

/// <summary>register Matrix / layers / loss / optimizer / LM step APIs on the module</summary>
void registerSentinelOps(nanobind::module_& m);

#endif // SENTINEL_PYTHON_BINDINGS_OPS_HPP
