#include <physica/cuda.h>

import std;
import physica.example.deformables.cloth.fast_projection;

int main() {
    try {
        physica::examples::cloth::fast_projection::Simulation simulation{};
        const physica::examples::cloth::fast_projection::Summary summary = simulation.run();
        std::println("algorithm=FastProjectionGoldenthal2007");
        std::println("constraints=all-topology-edge-distance linear_system=J*M^-1*J^T solver=matrix-free-PCG preconditioner=Jacobi");
        std::println("outer_iterations={} pcg_iterations={}", simulation.outer_iteration_count, simulation.pcg_iteration_count);
        std::println("mesh={}x{} width={:.6f} height={:.6f} anchors=top-corners fixed_particles=({}, {}) gravity=(0,{:.6f},0) mass={:.6f} wind=off", simulation.rows, simulation.columns, simulation.width, simulation.height, simulation.fixed_particles[0], simulation.fixed_particles[1], simulation.gravity_y, simulation.mass);
        std::println("frames={} dt={:.9f} physical_time={:.6f}", summary.frames, simulation.time_step, summary.physical_time);
        std::println("maximum_absolute_edge_constraint_error={:.9e}", summary.maximum_absolute_edge_constraint_error);
        std::println("maximum_stretch_ratio={:.9f}", summary.maximum_stretch_ratio);
        std::println("maximum_fixed_position_error={:.9e}", summary.maximum_fixed_position_error);
        std::println("probe_particle={} probe_position=({:.9f},{:.9f},{:.9f})", simulation.probe_particle, summary.probe_position.x, summary.probe_position.y, summary.probe_position.z);
        std::println("probe_velocity=({:.9f},{:.9f},{:.9f})", summary.probe_velocity.x, summary.probe_velocity.y, summary.probe_velocity.z);
        return 0;
    } catch (const std::exception& error) {
        std::println(std::cerr, "cloth Fast Projection example failed: {}", error.what());
        return 1;
    }
}
