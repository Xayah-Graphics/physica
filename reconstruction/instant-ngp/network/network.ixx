module;

#include <cublasLt.h>
#include <physica/cuda.h>

export module physica.reconstruction.instant_ngp.network;

import std;

export namespace physica::reconstruction::instant_ngp {
    enum class GridEncoding : std::uint8_t {
        multiresolution_hash,
    };

    enum class DirectionEncoding : std::uint8_t {
        spherical_harmonics,
    };

    enum class MlpActivation : std::uint8_t {
        relu,
    };

    struct GridShape final {
        GridEncoding encoding;
        std::uint32_t level_count;
        std::uint32_t features_per_level;
        std::uint32_t base_resolution;
        std::uint32_t resolution_scale;
        std::uint32_t log2_hashmap_size;

        constexpr bool operator==(const GridShape&) const = default;
    };

    struct DirectionShape final {
        DirectionEncoding encoding;
        std::uint32_t degree;

        constexpr bool operator==(const DirectionShape&) const = default;
    };

    struct MlpShape final {
        std::uint32_t width;
        std::uint32_t hidden_layer_count;
        std::uint32_t output_width;
        MlpActivation activation;

        constexpr bool operator==(const MlpShape&) const = default;
    };

    struct NetworkShape final {
        GridShape grid;
        DirectionShape direction;
        MlpShape density;
        MlpShape color;
        std::uint32_t training_batch_size;
        std::uint32_t inference_capacity;

        constexpr bool operator==(const NetworkShape&) const = default;
    };

    inline constexpr NetworkShape nerf_synthetic_network_shape{
        .grid =
            {
                .encoding           = GridEncoding::multiresolution_hash,
                .level_count        = 8u,
                .features_per_level = 4u,
                .base_resolution    = 16u,
                .resolution_scale   = 2u,
                .log2_hashmap_size  = 19u,
            },
        .direction =
            {
                .encoding = DirectionEncoding::spherical_harmonics,
                .degree   = 4u,
            },
        .density =
            {
                .width              = 64u,
                .hidden_layer_count = 1u,
                .output_width       = 16u,
                .activation         = MlpActivation::relu,
            },
        .color =
            {
                .width              = 64u,
                .hidden_layer_count = 2u,
                .output_width       = 16u,
                .activation         = MlpActivation::relu,
            },
        .training_batch_size = 1u << 18u,
        .inference_capacity  = (1u << 18u) * 16u,
    };

    struct Sample final {
        std::array<float, 3> position{};
        float step = 0.0F;
        std::array<float, 3> direction{};
    };

    static_assert(sizeof(Sample) == 7uz * sizeof(float));

    struct DeviceSamples final {
        const Sample* data  = nullptr;
        std::uint32_t count = 0u;
    };

    struct NetworkOutput final {
        const std::uint16_t* data = nullptr;
        std::uint32_t count       = 0u;
    };

    struct NetworkGradients final {
        const std::uint16_t* data = nullptr;
    };

    struct NetworkState final {
        std::vector<float> parameters;
        std::vector<float> first_moments;
        std::vector<float> second_moments;
        std::vector<std::uint32_t> parameter_steps;
    };

    struct NetworkDeviceState final {
        std::span<const std::uint16_t> hash_grid;
        std::span<const std::uint16_t> density_input;
        std::span<const std::uint16_t> density_hidden;
        std::span<const std::uint16_t> density_output;
        std::span<const std::uint16_t> color_input;
        std::span<const std::uint16_t> color_hidden;
        std::span<const std::uint16_t> color_output;
    };

    template <NetworkShape Shape>
    struct Network final {
        static_assert(Shape.grid.encoding == GridEncoding::multiresolution_hash, "Only multiresolution hash-grid encoding is implemented.");
        static_assert(Shape.grid.level_count == 8u, "Only the 8-level grid CUDA specialization is implemented.");
        static_assert(Shape.grid.features_per_level == 4u, "Only four grid features per level are implemented.");
        static_assert(Shape.grid.base_resolution == 16u, "Only base grid resolution 16 is implemented.");
        static_assert(Shape.grid.resolution_scale == 2u, "Only grid resolution scale 2 is implemented.");
        static_assert(Shape.grid.log2_hashmap_size == 19u, "Only a 2^19-entry hash map is implemented.");
        static_assert(Shape.direction.encoding == DirectionEncoding::spherical_harmonics, "Only spherical-harmonics direction encoding is implemented.");
        static_assert(Shape.direction.degree == 4u, "Only degree-4 spherical harmonics are implemented.");
        static_assert(Shape.density.width == 64u, "Only width-64 density MLPs are implemented.");
        static_assert(Shape.density.hidden_layer_count == 1u, "Only one density hidden layer is implemented.");
        static_assert(Shape.density.output_width == 16u, "Only density output width 16 is implemented.");
        static_assert(Shape.density.activation == MlpActivation::relu, "Only ReLU density MLPs are implemented.");
        static_assert(Shape.color.width == 64u, "Only width-64 color MLPs are implemented.");
        static_assert(Shape.color.hidden_layer_count == 2u, "Only two color hidden layers are implemented.");
        static_assert(Shape.color.output_width == 16u, "Only color output width 16 is implemented.");
        static_assert(Shape.color.activation == MlpActivation::relu, "Only ReLU color MLPs are implemented.");
        static_assert(Shape.training_batch_size == 1u << 18u, "Only a 2^18 training batch is implemented.");
        static_assert(Shape.inference_capacity == (1u << 18u) * 16u, "Only the current inference capacity is implemented.");

        Network(::cuda::stream_ref stream, std::uint32_t seed);
        ~Network() noexcept;

        Network(const Network&)            = delete;
        Network& operator=(const Network&) = delete;
        Network(Network&&)                 = delete;
        Network& operator=(Network&&)      = delete;

        NetworkOutput infer(DeviceSamples samples);
        NetworkOutput infer_density(DeviceSamples samples);
        void forward(DeviceSamples samples);
        void backward(DeviceSamples samples, NetworkGradients gradients);
        void step();
        NetworkState download() const;
        void upload(const NetworkState& state);
        [[nodiscard]] NetworkDeviceState device_state() const noexcept;

    private:
        static consteval std::uint32_t make_grid_entry_count() {
            std::uint32_t result                  = 0u;
            std::uint32_t resolution              = Shape.grid.base_resolution;
            const std::uint64_t hashmap_size      = 1ull << Shape.grid.log2_hashmap_size;
            const std::uint64_t maximum_positions = (std::numeric_limits<std::uint32_t>::max)() / 2ull;
            for (std::uint32_t level = 0u; level < Shape.grid.level_count; ++level) {
                const std::uint64_t dense   = static_cast<std::uint64_t>(resolution) * resolution * resolution;
                const std::uint64_t bounded = dense < maximum_positions ? dense : maximum_positions;
                result += static_cast<std::uint32_t>((std::min) (((bounded + 7u) / 8u) * 8u, hashmap_size));
                resolution *= Shape.grid.resolution_scale;
            }
            return result;
        }

        inline static constexpr std::uint32_t grid_output_width        = Shape.grid.level_count * Shape.grid.features_per_level;
        inline static constexpr std::uint32_t direction_output_width   = Shape.direction.degree * Shape.direction.degree;
        inline static constexpr std::uint32_t color_input_width        = Shape.density.output_width + direction_output_width;
        inline static constexpr std::uint32_t grid_entry_count         = make_grid_entry_count();
        inline static constexpr std::uint32_t density_input_offset     = 0u;
        inline static constexpr std::uint32_t density_input_count      = Shape.density.width * grid_output_width;
        inline static constexpr std::uint32_t density_hidden_offset    = density_input_offset + density_input_count;
        inline static constexpr std::uint32_t density_hidden_count     = (Shape.density.hidden_layer_count - 1u) * Shape.density.width * Shape.density.width;
        inline static constexpr std::uint32_t density_output_offset    = density_hidden_offset + density_hidden_count;
        inline static constexpr std::uint32_t density_output_count     = Shape.density.output_width * Shape.density.width;
        inline static constexpr std::uint32_t color_input_offset       = density_output_offset + density_output_count;
        inline static constexpr std::uint32_t color_input_count        = Shape.color.width * color_input_width;
        inline static constexpr std::uint32_t color_hidden_offset      = color_input_offset + color_input_count;
        inline static constexpr std::uint32_t color_hidden_count       = (Shape.color.hidden_layer_count - 1u) * Shape.color.width * Shape.color.width;
        inline static constexpr std::uint32_t color_output_offset      = color_hidden_offset + color_hidden_count;
        inline static constexpr std::uint32_t color_output_count       = Shape.color.output_width * Shape.color.width;
        inline static constexpr std::uint32_t grid_offset              = color_output_offset + color_output_count;
        inline static constexpr std::uint32_t grid_count               = grid_entry_count * Shape.grid.features_per_level;
        inline static constexpr std::size_t parameter_count            = static_cast<std::size_t>(grid_offset) + grid_count;
        inline static constexpr std::size_t cublaslt_matmul_plan_count = 7uz;

        struct CublasLtMatmulPlans final {
            std::array<cublasLtMatmulDesc_t, cublaslt_matmul_plan_count> operation_descriptors{};
            std::array<cublasLtMatrixLayout_t, cublaslt_matmul_plan_count> a_descriptors{};
            std::array<cublasLtMatrixLayout_t, cublaslt_matmul_plan_count> b_descriptors{};
            std::array<cublasLtMatrixLayout_t, cublaslt_matmul_plan_count> output_descriptors{};
            std::array<cublasLtMatmulAlgo_t, cublaslt_matmul_plan_count> algorithms{};

            ~CublasLtMatmulPlans() noexcept;
        };

        struct CublasLtDeleter final {
            void operator()(cublasLtContext* handle) const noexcept;
        };

        ::cuda::stream_ref stream;
        std::unique_ptr<cublasLtContext, CublasLtDeleter> cublaslt_handle;
        CublasLtMatmulPlans cublaslt_matmul_plans;
        ::cuda::device_buffer<float> parameters_full_precision;
        ::cuda::device_buffer<std::uint16_t> parameters;
        ::cuda::device_buffer<std::uint16_t> gradients;
        ::cuda::device_buffer<float> first_moments;
        ::cuda::device_buffer<float> second_moments;
        ::cuda::device_buffer<std::uint32_t> parameter_steps;
        ::cuda::device_buffer<std::uint16_t> density_input;
        ::cuda::device_buffer<std::uint16_t> field_features;
        ::cuda::device_buffer<std::uint16_t> output;
        ::cuda::device_buffer<std::uint16_t> color_output_gradients;
        ::cuda::device_buffer<std::uint16_t> color_input_gradients;
        ::cuda::device_buffer<std::uint16_t> density_input_gradients;
        ::cuda::device_buffer<std::uint16_t> density_forward_hidden;
        ::cuda::device_buffer<std::uint16_t> color_forward_hidden;
        ::cuda::device_buffer<std::uint16_t> density_backward_hidden;
        ::cuda::device_buffer<std::uint16_t> color_backward_hidden;
        ::cuda::device_buffer<std::uint8_t> cublaslt_workspace;
    };

    extern template struct Network<nerf_synthetic_network_shape>;
} // namespace physica::reconstruction::instant_ngp
