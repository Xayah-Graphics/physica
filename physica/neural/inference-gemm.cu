#include "inference-kernels.h"
#include <cutlass/epilogue/thread/linear_combination.h>
#include <device/dual_gemm.h>
#include <stdexcept>

namespace physica::neural::kernels {
    struct GeGLU : cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, cutlass::half_t, float> {
        CUTLASS_HOST_DEVICE explicit GeGLU(const Params& parameters) : cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, cutlass::half_t, float>{parameters} {}
    };
} // namespace physica::neural::kernels

// Consume CUTLASS's FP32 accumulator fragments before any FP16 conversion.
namespace cutlass::epilogue::threadblock {
    template <class Shape, class Mma, int Partitions, class Output, class AccumulatorIterator, class WarpIterator, class SharedIterator, class Op0, class Op1, class Padding, int Fragments, int Unroll>
    struct DualEpilogue<Shape, Mma, Partitions, Output, AccumulatorIterator, WarpIterator, SharedIterator, Op0, Op1, physica::neural::kernels::GeGLU, Padding, false, false, Fragments, Unroll> : DualEpilogue<Shape, Mma, Partitions, Output, AccumulatorIterator, WarpIterator, SharedIterator, Op0, Op1, thread::LinearCombination<half_t, 8, half_t, float>, Padding, false, false, Fragments, Unroll> {
        struct SharedStorage : DualEpilogue<Shape, Mma, Partitions, Output, AccumulatorIterator, WarpIterator, SharedIterator, Op0, Op1, thread::LinearCombination<half_t, 8, half_t, float>, Padding, false, false, Fragments, Unroll>::SharedStorage {};
        SharedIterator load0;
        SharedIterator load1;
        WarpIterator store0;
        WarpIterator store1;

        CUTLASS_DEVICE DualEpilogue(SharedStorage& shared, int thread, int warp, int lane) : DualEpilogue<Shape, Mma, Partitions, Output, AccumulatorIterator, WarpIterator, SharedIterator, Op0, Op1, thread::LinearCombination<half_t, 8, half_t, float>, Padding, false, false, Fragments, Unroll>{shared, thread, warp, lane}, load0{shared.reference(0), thread}, load1{shared.reference(1), thread}, store0{shared.reference(0), lane}, store1{shared.reference(1), lane} {
            constexpr int Rows = Shape::kM / Mma::Shape::kM;
            const MatrixCoord offset{warp % Rows, warp / Rows};
            store0.add_tile_offset(offset);
            store1.add_tile_offset(offset);
        }

        template <class Accumulator>
        CUTLASS_DEVICE void operator()(const Op0&, const Op1&, const physica::neural::kernels::GeGLU&, Output, Output, Output destination, const Accumulator& value, const Accumulator& gate, Output source[2], bool) {
            AccumulatorIterator value_iterator{value};
            AccumulatorIterator gate_iterator{gate};
            CUTLASS_PRAGMA_UNROLL
            for (int iteration = 0; iteration < Output::kIterations; ++iteration) {
                typename Output::Fragment bias0, bias1, result;
                bias0.clear();
                bias1.clear();
                source[0].load(bias0);
                source[1].load(bias1);
                ++source[0];
                ++source[1];
                typename AccumulatorIterator::Fragment warp0, warp1;
                value_iterator.load(warp0);
                gate_iterator.load(warp1);
                ++value_iterator;
                ++gate_iterator;
                __syncthreads();
                store0.store(warp0);
                store1.store(warp1);
                __syncthreads();
                typename SharedIterator::Fragment values, gates;
                load0.load(values);
                load1.load(gates);
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < Output::Fragment::kElements; ++i) {
                    const float x = gates[i] + float(bias1[i]);
                    const float y = values[i] + float(bias0[i]);
                    result[i]     = half_t(y * (0.5F * x * (1.0F + erff(x * 0.7071067811865476F))));
                }
                destination.store(result);
                ++destination;
            }
        }
    };
} // namespace cutlass::epilogue::threadblock

namespace physica::neural::kernels {
    template <int M, int N, int K, int WarpM, int WarpN, int Stages>
    void geglu(const ::cuda::stream_ref stream, void* output, const void* input, const void* weight, const void* bias, const int rows, const int width) {
        cutlass::gemm::device::DualGemm<cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::half_t, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<M, N, K>, cutlass::gemm::GemmShape<WarpM, WarpN, K>, cutlass::gemm::GemmShape<16, 8, 16>, cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float, cutlass::epilogue::thread::ScaleType::NoBetaScaling>, cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float, cutlass::epilogue::thread::ScaleType::NoBetaScaling>, GeGLU, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages, false, false, false, 8, 8> operation;
        const int expanded  = width * 4;
        const auto* weights = static_cast<const cutlass::half_t*>(weight);
        const auto* biases  = static_cast<const cutlass::half_t*>(bias);
        const typename decltype(operation)::Arguments arguments{cutlass::gemm::DualGemmMode::kGemm, {rows, expanded, width}, {static_cast<const cutlass::half_t*>(input), width}, {weights, width}, {biases, 0}, {nullptr, expanded}, {weights + static_cast<std::size_t>(expanded) * width, width}, {biases + expanded, 0}, {nullptr, expanded}, {static_cast<cutlass::half_t*>(output), expanded}, {1.0F}, {1.0F}, {}};
        if (operation(arguments, nullptr, stream.get()) != cutlass::Status::kSuccess) throw std::runtime_error{"SDXL CUTLASS GEGLU launch failed"};
    }

    void linear_geglu(const ::cuda::stream_ref stream, void* output, const void* input, const void* weight, const void* bias, const int rows, const int width, const int configuration) {
        switch (configuration) {
        case 0: geglu<128, 64, 32, 64, 32, 3>(stream, output, input, weight, bias, rows, width); break;
        case 1: geglu<64, 64, 32, 32, 32, 4>(stream, output, input, weight, bias, rows, width); break;
        case 2: geglu<128, 64, 64, 64, 32, 3>(stream, output, input, weight, bias, rows, width); break;
        case 3: geglu<64, 128, 32, 32, 64, 4>(stream, output, input, weight, bias, rows, width); break;
        default: throw std::runtime_error{"SDXL GEGLU cache contains an unknown configuration"};
        }
    }
} // namespace physica::neural::kernels
