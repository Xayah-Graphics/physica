module;
#include <physica/cuda.h>

module physica.generative.sdxl.workspace;

import std;

namespace physica::generative::sdxl {
    WorkspaceLayout::WorkspaceLayout(const std::size_t normalized_bytes, const std::size_t hidden_bytes, const std::size_t output_bytes, const std::size_t shortcut_bytes, const std::size_t combined_bytes) : normalized{0}, hidden{(normalized_bytes + 255) & ~255uz}, output{hidden + ((hidden_bytes + 255) & ~255uz)}, shortcut{output + ((output_bytes + 255) & ~255uz)}, combined{shortcut + ((shortcut_bytes + 255) & ~255uz)}, statistics{combined + ((combined_bytes + 255) & ~255uz)}, bytes{statistics + 2uz * 32 * 1024 * 3 * sizeof(float)} {}

    Workspace WorkspaceLayout::view(std::byte* memory) const {
        return {{memory + normalized}, {memory + hidden}, {memory + output}, {memory + shortcut}, {memory + combined}, reinterpret_cast<float*>(memory + statistics)};
    }

    ClipWorkspaceLayout::ClipWorkspaceLayout(const std::size_t tokens) : WorkspaceLayout{tokens * 1280 * 2, tokens * 5120 * 2, tokens * 1280 * 2, 1280 * 2, 1280 * 2} {}

    UNetWorkspaceLayout::UNetWorkspaceLayout(const int height, const int width, const int steps) : WorkspaceLayout{std::size_t(height) * width * 2 * 960 * 2, std::max(std::size_t(height) * width * 2 * 640 * 2, std::size_t(steps) * 1280 * 2), 0, std::max(std::size_t(height) * width * 2 * 640 * 2, std::size_t(steps) * 1280 * 2), std::max(std::size_t(height) * width * 2 * 960 * 2, std::size_t(steps) * 2 * 1280 * 2)} {
        // Transformer output and ResNet shortcut never overlap.
        output = shortcut;
    }

    VAEWorkspaceLayout::VAEWorkspaceLayout(const int height, const int width) : WorkspaceLayout{std::size_t(height) * width * 256 * 2, std::size_t(height) * width * 128 * 2, 0, std::size_t(height) * width * 128 * 2, 0} {
        // Resizing and normalization are separate phases; attention precedes
        // the channel-changing ResNets that need the shortcut storage.
        combined = normalized;
        output   = shortcut;
    }
} // namespace physica::generative::sdxl
