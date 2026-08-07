cmake_minimum_required(VERSION 3.24)

project(sentinel
    VERSION 0.1.0
    DESCRIPTION "C++/CUDA full-train causal language model framework"
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Prefer the same host compiler nvcc will use (important on Linux).
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT MSVC)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "Host compiler for nvcc")
endif()

# Fat binary for GeForce 20/30/40/50 (Turing → Blackwell). CUDA 13+ floor is sm_75.
# Override for faster local builds, e.g. -DSENTINEL_CUDA_ARCHITECTURES=native
set(SENTINEL_CUDA_ARCHITECTURES "75;80;86;89;120" CACHE STRING
    "CUDA architectures (semicolon-separated). Use native for host GPU only.")
set(CMAKE_CUDA_ARCHITECTURES "${SENTINEL_CUDA_ARCHITECTURES}")

option(SENTINEL_BUILD_SHARED "Build sentinel as a shared library" OFF)
option(SENTINEL_BUILD_DEMO "Build the main demo harness (sentinel executable)" ON)
option(SENTINEL_BUILD_EXAMPLES "Build example programs" ON)
option(SENTINEL_BUILD_PYTHON "Build Python nanobind module (sentinel._core)" OFF)
option(SENTINEL_INSTALL "Generate install/export rules" ON)

# pip/scikit-build-core sets SKBUILD; prefer a lean wheel (lib + Python only).
if(DEFINED SKBUILD)
    set(SENTINEL_BUILD_DEMO OFF CACHE BOOL "" FORCE)
    set(SENTINEL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SENTINEL_BUILD_PYTHON ON CACHE BOOL "" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)
find_package(Threads REQUIRED)
include(GNUInstallDirs)

set(SENTINEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/sentinel")

set(SENTINEL_CXX_SOURCES
    "${SENTINEL_ROOT}/NeuralNet/Activations/ReLU.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Activations/SiLU.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Activations/Softmax.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Data/ClassificationDataset.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Data/DatasetSplit.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Data/JsonlLoader.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Data/TextRowReader.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Data/ArrowChunkReader.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Data/LanguageModelChunkSource.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Data/LanguageModelDataset.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Initializers/UniformInit.cpp"
    "${SENTINEL_ROOT}/NeuralNet/IO/SafeTensors.cpp"
    "${SENTINEL_ROOT}/NeuralNet/IO/HuggingFaceConfig.cpp"
    "${SENTINEL_ROOT}/NeuralNet/IO/HuggingFaceWeights.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Utils/TextUtil.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Layers/Dense.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Layers/Dropout.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Layers/CausalSelfAttention.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Layers/Embedding.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Layers/FeedForward.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Layers/RMSNorm.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Layers/RotaryEmbedding.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Layers/TransformerBlock.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Layers/MeanPool.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Losses/CrossEntropy.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Losses/MSE.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Math/Matrix.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Network/LanguageModel.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Network/Sequential.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Optimizers/Adam.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Optimizers/SGD.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Optimizers/Spulse.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Tokenizer/BPETokenizer.cpp"
    "${SENTINEL_ROOT}/NeuralNet/Tokenizer/HfTokenizer.cpp"
)

set(SENTINEL_CUDA_SOURCES
    "${SENTINEL_ROOT}/NeuralNet/Cuda/CudaMatmul.cu"
    "${SENTINEL_ROOT}/NeuralNet/Cuda/CudaAmp.cu"
    "${SENTINEL_ROOT}/NeuralNet/Cuda/CudaOps.cu"
    "${SENTINEL_ROOT}/NeuralNet/Cuda/CudaFeedForward.cu"
    "${SENTINEL_ROOT}/NeuralNet/Cuda/CudaRMSNorm.cu"
    "${SENTINEL_ROOT}/NeuralNet/Cuda/CudaCausalSelfAttention.cu"
    "${SENTINEL_ROOT}/NeuralNet/Cuda/CudaFlashAttention.cu"
    "${SENTINEL_ROOT}/NeuralNet/Cuda/CudaAdam.cu"
    "${SENTINEL_ROOT}/NeuralNet/Cuda/CudaSbao.cu"
    "${SENTINEL_ROOT}/NeuralNet/Cuda/CudaMuon.cu"
    "${SENTINEL_ROOT}/NeuralNet/Cuda/CudaSPULSE.cu"
    "${SENTINEL_ROOT}/NeuralNet/Cuda/CudaKvCache.cu"
    "${SENTINEL_ROOT}/NeuralNet/Cuda/CudaTransformerBlock.cu"
    "${SENTINEL_ROOT}/NeuralNet/Cuda/CudaLanguageModel.cu"
)

if(SENTINEL_BUILD_SHARED)
    add_library(sentinel SHARED ${SENTINEL_CXX_SOURCES} ${SENTINEL_CUDA_SOURCES})
else()
    add_library(sentinel STATIC ${SENTINEL_CXX_SOURCES} ${SENTINEL_CUDA_SOURCES})
endif()
add_library(Sentinel::sentinel ALIAS sentinel)

set_target_properties(sentinel PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
)

target_include_directories(sentinel PUBLIC
    $<BUILD_INTERFACE:${SENTINEL_ROOT}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(sentinel PUBLIC
    $<$<CONFIG:Debug>:_DEBUG>
    $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
)

# CUDA + host deps are PUBLIC so static consumers pick them up via Sentinel::sentinel.
target_link_libraries(sentinel PUBLIC
    CUDA::cudart_static
    CUDA::cublasLt
    CUDA::cublas
    Threads::Threads
)

if(UNIX AND NOT APPLE)
    target_link_libraries(sentinel PUBLIC ${CMAKE_DL_LIBS} rt)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(sentinel PUBLIC OpenMP::OpenMP_CXX)
    if(MSVC)
        target_compile_options(sentinel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/openmp>)
    else()
        separate_arguments(_sentinel_omp_cuda_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
        foreach(_omp_flag IN LISTS _sentinel_omp_cuda_flags)
            if(_omp_flag)
                target_compile_options(sentinel PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_omp_flag}>")
            endif()
        endforeach()
    endif()
else()
    message(WARNING "OpenMP not found — host parallel paths will be single-threaded")
endif()

if(MSVC)
    target_compile_options(sentinel PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3 /EHsc /MP>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>
    )
else()
    target_compile_options(sentinel PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>
    )
endif()

function(sentinel_apply_common_exe_settings target_name)
    target_link_libraries(${target_name} PRIVATE Sentinel::sentinel)
    target_compile_definitions(${target_name} PRIVATE _CONSOLE)
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
        VS_DEBUGGER_WORKING_DIRECTORY "${SENTINEL_ROOT}"
    )
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W3 /EHsc)
        set_property(TARGET ${target_name} PROPERTY WIN32_EXECUTABLE FALSE)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra)
    endif()
endfunction()

if(SENTINEL_BUILD_DEMO)
    add_executable(sentinel_demo "${SENTINEL_ROOT}/main.cpp")
    set_target_properties(sentinel_demo PROPERTIES OUTPUT_NAME "sentinel")
    sentinel_apply_common_exe_settings(sentinel_demo)
endif()

if(SENTINEL_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(SENTINEL_BUILD_PYTHON)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(sentinel_core NB_STATIC
        "${CMAKE_CURRENT_SOURCE_DIR}/python/sentinel/_core.cpp"
    )
    set_target_properties(sentinel_core PROPERTIES OUTPUT_NAME "_core")
    target_link_libraries(sentinel_core PRIVATE Sentinel::sentinel)
    # Extension lives inside the Python package directory (python/sentinel/).
    if(DEFINED SKBUILD)
        install(TARGETS sentinel_core
            LIBRARY DESTINATION sentinel
            RUNTIME DESTINATION sentinel
            ARCHIVE DESTINATION sentinel
        )
    else()
        set_target_properties(sentinel_core PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/sentinel"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/sentinel"
        )
        add_custom_command(TARGET sentinel_core POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/python/sentinel/__init__.py"
                "$<TARGET_FILE_DIR:sentinel_core>/__init__.py"
            COMMENT "Copy python/sentinel/__init__.py next to _core extension"
        )
    endif()
endif()

if(SENTINEL_INSTALL AND NOT DEFINED SKBUILD)
    install(TARGETS sentinel
        EXPORT SentinelTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    if(SENTINEL_BUILD_DEMO)
        install(TARGETS sentinel_demo RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()

    install(DIRECTORY "${SENTINEL_ROOT}/NeuralNet"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING
            PATTERN "*.hpp"
            PATTERN "*.h"
    )

    include(CMakePackageConfigHelpers)
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SentinelConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/SentinelConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Sentinel"
    )
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/SentinelConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(EXPORT SentinelTargets
        FILE SentinelTargets.cmake
        NAMESPACE Sentinel::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Sentinel"
    )
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/SentinelConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/SentinelConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Sentinel"
    )
endif()

if(SENTINEL_BUILD_EXAMPLES AND SENTINEL_INSTALL AND NOT DEFINED SKBUILD)
    # examples install handled in examples/CMakeLists.txt
endif()

message(STATUS "sentinel ${PROJECT_VERSION} platform: ${CMAKE_SYSTEM_NAME}")
if(SENTINEL_BUILD_SHARED)
    message(STATUS "sentinel library type: shared")
else()
    message(STATUS "sentinel library type: static")
endif()
message(STATUS "sentinel CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION} (${CUDAToolkit_LIBRARY_ROOT})")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER}")
message(STATUS "Build demo=${SENTINEL_BUILD_DEMO} examples=${SENTINEL_BUILD_EXAMPLES} python=${SENTINEL_BUILD_PYTHON}")
