llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-hybrid-iswa.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampler.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            triattention.c
            triattention-runtime.c
            triattention-backend.c
            triattention-bridge.cpp
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

# TriAttention KV-cache scoring parallelizes across KV heads via OpenMP.
# Optional: if OpenMP is unavailable the pragmas are ignored and scoring runs
# serially (the prior behavior), so this never breaks the build.
find_package(OpenMP QUIET)
if (OpenMP_C_FOUND)
    target_link_libraries(llama PRIVATE OpenMP::OpenMP_C)
endif()

# GPU TriAttention scorer (ggml/src/ggml-cuda/triattention-score.cu).  The
# kernel entry points are resolved through the PUBLIC link chain to ggml-cuda.
# GGML_USE_CUDA is only defined for HIP builds, so use a dedicated macro to gate
# the host call sites in triattention-runtime.c for pure-CUDA builds.
if (GGML_CUDA)
    target_compile_definitions(llama PRIVATE GGML_TRIA_CUDA)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
