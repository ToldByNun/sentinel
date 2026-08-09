#ifndef SENTINEL_PYTHON_BINDINGS_DATA_HPP
#define SENTINEL_PYTHON_BINDINGS_DATA_HPP

#include <nanobind/nanobind.h>

/// <summary>register dataset / streaming / safetensors / Sequential APIs</summary>
void registerSentinelData(nanobind::module_& m);

#endif // SENTINEL_PYTHON_BINDINGS_DATA_HPP
