#include <physica/cuda.h>

import std;
import physica.example.deformables.cloth.fast_mass_spring;

int main() {
    try {
        physica::examples::cloth::fast_mass_spring::Simulation simulation{};
        const physica::examples::cloth::fast_mass_spring::Summary summary = simulation.run();

        std::println("algorithm=FastMassSpringLiu2013");
        std::println("springs=all-topology-edges local_projection=rest-length global_system=M/h^2+kL solver=cuDSS-SPD rhs_layout=3-column-major deterministic_solves=3x-single-RHS");
        std::println("global_iterations={} spring_stiffness={:.6f}", simulation.global_iteration_count, simulation.spring_stiffness);
        std::println("mesh={}x{} width={:.6f} height={:.6f} anchors=top-corners fixed_particles=({}, {}) gravity=(0,{:.6f},0) mass={:.6f}", simulation.rows, simulation.columns, simulation.width, simulation.height, simulation.fixed_particles[0], simulation.fixed_particles[1], simulation.gravity_y, simulation.mass);
        std::println("initial_perturbation_z={:.6f} profile=sin(row)*sin(2*column)", simulation.initial_perturbation);
        std::println("frames={} dt={:.9f} physical_time={:.6f}", summary.frames, simulation.time_step, summary.physical_time);
        std::println("maximum_absolute_edge_length_error={:.9e}", summary.maximum_absolute_edge_length_error);
        std::println("maximum_stretch_ratio={:.9f}", summary.maximum_stretch_ratio);
        std::println("maximum_fixed_position_error={:.9e}", summary.maximum_fixed_position_error);
        std::println("probe_particle={} probe_position=({:.9f},{:.9f},{:.9f})", simulation.probe_particle, summary.probe_position.x, summary.probe_position.y, summary.probe_position.z);
        std::println("probe_velocity=({:.9f},{:.9f},{:.9f})", summary.probe_velocity.x, summary.probe_velocity.y, summary.probe_velocity.z);
        return 0;
    } catch (const std::exception& error) {
        std::println(std::cerr, "cloth Fast Mass-Spring example failed: {}", error.what());
        return 1;
    }
}
