#include <physica/cuda.h>

import std;
import physica.example.deformables.cloth.xpbd;

int main() {
    try {
        physica::examples::cloth::xpbd::Simulation simulation{};
        const physica::examples::cloth::xpbd::Summary summary = simulation.run();
        std::println("algorithm=XPBD");
        std::println("constraint=compliant-distance projection=colored-gauss-seidel iterations={}", simulation.projection_iterations);
        std::println("compliance stretch={:.9e} bending={:.9e}", simulation.stretch_compliance, simulation.bending_compliance);
        std::println("mesh={}x{} anchors=top-corners gravity_y={:.3f} mass={:.6f}", simulation.rows, simulation.columns, simulation.gravity_y, simulation.mass);
        std::println("frames={} dt={:.9f} physical_time={:.6f}", summary.frames, simulation.time_step, summary.physical_time);
        std::println("maximum_stretch_ratio={:.9f}", summary.maximum_stretch_ratio);
        std::println("maximum_fixed_position_error={:.9e}", summary.maximum_fixed_position_error);
        std::println("final_maximum_absolute_lambda={:.9e}", summary.final_maximum_absolute_lambda);
        std::println("probe_position=({:.9f},{:.9f},{:.9f})", summary.probe_position.x, summary.probe_position.y, summary.probe_position.z);
        std::println("probe_velocity=({:.9f},{:.9f},{:.9f})", summary.probe_velocity.x, summary.probe_velocity.y, summary.probe_velocity.z);
        return 0;
    } catch (const std::exception& error) {
        std::println(std::cerr, "cloth XPBD example failed: {}", error.what());
        return 1;
    }
}
