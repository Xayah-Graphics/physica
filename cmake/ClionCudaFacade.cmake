# Temporary CLion workaround for CCCL indexing.
set(PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY "${PROJECT_BINARY_DIR}/generated/clion-cuda/include")

file(MAKE_DIRECTORY "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}/physica")

file(
        CONFIGURE
        OUTPUT "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}/physica/cuda.h"
        CONTENT [=[
#ifndef PHYSICA_CUDA_H
#define PHYSICA_CUDA_H

#ifdef __JETBRAINS_IDE__
#include <physica/cuda_clion.h>
#else
#include <cuda/__functional/call_or.h>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/std/span>
#include <cuda/stream>
#endif

#endif
]=]
        @ONLY
        NEWLINE_STYLE UNIX
)

file(
        CONFIGURE
        OUTPUT "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}/physica/cuda_stream.h"
        CONTENT [=[
#ifndef PHYSICA_CUDA_STREAM_H
#define PHYSICA_CUDA_STREAM_H

#if defined(__JETBRAINS_IDE__) && !defined(__CUDACC__)
#include <physica/cuda_clion.h>
#else
#include <cuda/stream>
#endif

#endif
]=]
        @ONLY
        NEWLINE_STYLE UNIX
)

file(
        CONFIGURE
        OUTPUT "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}/physica/cuda_clion.h"
        CONTENT [=[
#ifndef PHYSICA_CUDA_CLION_H
#define PHYSICA_CUDA_CLION_H

#include <cstddef>

struct CUstream_st;

namespace cuda {
    struct device_ref final {};

    struct devices_view final {
        [[nodiscard]] device_ref operator[](::std::size_t index) const;
    };

    inline constexpr devices_view devices{};

    struct no_init_t final {};
    inline constexpr no_init_t no_init{};

    struct memory_pool_ref final {};

    [[nodiscard]] memory_pool_ref device_default_memory_pool(device_ref device);
    [[nodiscard]] memory_pool_ref pinned_default_memory_pool();

    struct stream_ref {
        stream_ref() = default;

        [[nodiscard]] CUstream_st* get() const;
        [[nodiscard]] device_ref device() const;
        void sync() const;
    };

    struct stream final : stream_ref {
        explicit stream(device_ref device);
    };

    template<class Type>
    struct device_buffer final {
        device_buffer() = default;

        template<class... Arguments>
        device_buffer(Arguments&&... arguments);

        [[nodiscard]] Type* data();
        [[nodiscard]] const Type* data() const;
        [[nodiscard]] ::std::size_t size() const;
        [[nodiscard]] bool empty() const;
        [[nodiscard]] Type* begin();
        [[nodiscard]] const Type* begin() const;
        [[nodiscard]] Type* end();
        [[nodiscard]] const Type* end() const;
        Type& operator[](::std::size_t index);
        const Type& operator[](::std::size_t index) const;
    };

    template<class Type>
    struct host_buffer final {
        host_buffer() = default;

        template<class... Arguments>
        host_buffer(Arguments&&... arguments);

        [[nodiscard]] Type* data();
        [[nodiscard]] const Type* data() const;
        [[nodiscard]] ::std::size_t size() const;
        [[nodiscard]] bool empty() const;
        [[nodiscard]] Type* begin();
        [[nodiscard]] const Type* begin() const;
        [[nodiscard]] Type* end();
        [[nodiscard]] const Type* end() const;
        Type& operator[](::std::size_t index);
        const Type& operator[](::std::size_t index) const;
    };

    template<class... Arguments>
    void copy_bytes(Arguments&&... arguments);

    template<class... Arguments>
    void fill_bytes(Arguments&&... arguments);

    namespace std {
        template<class Type>
        struct span final {
            span(Type* data, ::std::size_t size);

            [[nodiscard]] Type* data() const;
            [[nodiscard]] ::std::size_t size() const;
        };

        template<class Type>
        span(Type* data, ::std::size_t size) -> span<Type>;
    }
}

#endif
]=]
        @ONLY
        NEWLINE_STYLE UNIX
)
