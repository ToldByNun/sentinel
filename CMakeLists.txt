cmake_minimum_required(VERSION 3.24)

project(sentinel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

# Prefer the same host compiler nvcc will use (important on Linux).
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT MSVC)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "Host compiler for nvcc")
endif()

# Fat binary for GeForce 20/30/40/50 (Turing → Blackwell). CUDA 13+ floor is sm_75.
# Override for faster local builds, e.g. -DSENTINEL_CUDA_ARCHITECTURES=native
set(SENTINEL_CUDA_ARCHITECTURES "75;80;86;89;120" CACHE STRING
    "CUDA architectures (semicolon-separated). Use native for host GPU only.")
set(CMAKE_CUDA_ARCHITECTURES "${SENTINEL_CUDA_ARCHITECTURES}")

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)
find_package(Threads REQUIRED)

set(SENTINEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/sentinel")

set(SENTINEL_CXX_SOURCES
    "${SENTINEL_ROOT}/main.cpp"
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
    "${SENTINEL_ROOT}/NeuralNet/Tokenizer/BPETokenizer.cpp"
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
    "${SENTINEL_ROOT}/NeuralNet/Cuda/CudaMuon.cu"
    "${SENTINEL_ROOT}/NeuralNet/Cuda/CudaKvCache.cu"
    "${SENTINEL_ROOT}/NeuralNet/Cuda/CudaTransformerBlock.cu"
    "${SENTINEL_ROOT}/NeuralNet/Cuda/CudaLanguageModel.cu"
)

add_executable(sentinel ${SENTINEL_CXX_SOURCES} ${SENTINEL_CUDA_SOURCES})

target_include_directories(sentinel PRIVATE "${SENTINEL_ROOT}")

target_compile_definitions(sentinel PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
    $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
    _CONSOLE
)

target_link_libraries(sentinel PRIVATE
    CUDA::cudart_static
    CUDA::cublasLt
    CUDA::cublas
    Threads::Threads
)

# Static cudart on Linux needs libdl + librt (and pthread via Threads above).
if(UNIX AND NOT APPLE)
    target_link_libraries(sentinel PRIVATE ${CMAKE_DL_LIBS} rt)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(sentinel PRIVATE OpenMP::OpenMP_CXX)
    if(MSVC)
        target_compile_options(sentinel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/openmp>)
    else()
        # Pass OpenMP to the host compiler inside nvcc (GCC/Clang: typically -fopenmp).
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
    set_property(TARGET sentinel PROPERTY WIN32_EXECUTABLE FALSE)
else()
    target_compile_options(sentinel PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>
    )
endif()

# Runtime expects cwd = sentinel/ for relative ../SERA-Data paths.
set_target_properties(sentinel PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
    VS_DEBUGGER_WORKING_DIRECTORY "${SENTINEL_ROOT}"
)

message(STATUS "sentinel platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "sentinel CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION} (${CUDAToolkit_LIBRARY_ROOT})")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER}")
