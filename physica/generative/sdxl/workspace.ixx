module;
#include <physica/cuda.h>

export module physica.generative.sdxl.workspace;

import std;
import physica.neural.inference_runtime;

export namespace physica::generative::sdxl {
    struct Workspace final {
        neural::TensorView normalized;
        neural::TensorView hidden;
        neural::TensorView output;
        neural::TensorView shortcut;
        neural::TensorView combined;
        float* statistics;
    };

    struct WorkspaceLayout {
        std::size_t normalized;
        std::size_t hidden;
        std::size_t output;
        std::size_t shortcut;
        std::size_t combined;
        std::size_t statistics;
        std::size_t bytes;

        WorkspaceLayout(std::size_t normalized_bytes, std::size_t hidden_bytes, std::size_t output_bytes, std::size_t shortcut_bytes, std::size_t combined_bytes);
        Workspace view(std::byte* memory) const;
    };

    struct ClipWorkspaceLayout final : WorkspaceLayout {
        explicit ClipWorkspaceLayout(std::size_t tokens);
    };

    struct UNetWorkspaceLayout final : WorkspaceLayout {
        UNetWorkspaceLayout(int height, int width, int steps);
    };

    struct VAEWorkspaceLayout final : WorkspaceLayout {
        VAEWorkspaceLayout(int height, int width);
    };
} // namespace physica::generative::sdxl
