#include <physica/cuda.h>

import std;
import physica.example.deformables.cloth.provot_strain_limiting;

int main() {
    try {
        physica::examples::cloth::provot_strain_limiting::Simulation simulation{};
        const physica::examples::cloth::provot_strain_limiting::Summary summary = simulation.run();
        std::println("algorithm=ProvotStrainLimit");
        std::println("projection=colored-gauss-seidel iterations={} maximum_stretch_ratio={:.6f}", simulation.projection_iterations, simulation.maximum_stretch_ratio);
        std::println("mesh={}x{} anchors=top-corners gravity_y={:.3f} stretch_stiffness={:.3f}", simulation.rows, simulation.columns, simulation.gravity_y, simulation.stretch_stiffness);
        std::println("frames={} dt={:.9f} physical_time={:.6f}", summary.frames, simulation.time_step, summary.physical_time);
        std::println("maximum_edge_stretch integrated={:.9f} projected={:.9f}", summary.integrated_maximum_stretch_ratio, summary.projected_maximum_stretch_ratio);
        std::println("probe_y initial={:.9f} final={:.9f}", summary.initial_probe_y, summary.final_probe_position.y);
        std::println("probe_position=({:.9f},{:.9f},{:.9f})", summary.final_probe_position.x, summary.final_probe_position.y, summary.final_probe_position.z);
        std::println("probe_velocity=({:.9f},{:.9f},{:.9f})", summary.final_probe_velocity.x, summary.final_probe_velocity.y, summary.final_probe_velocity.z);
        return 0;
    } catch (const std::exception& error) {
        std::println(std::cerr, "cloth Provot strain limiting example failed: {}", error.what());
        return 1;
    }
}
