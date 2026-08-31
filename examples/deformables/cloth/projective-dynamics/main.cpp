#include <physica/cuda.h>

import std;
import physica.example.deformables.cloth.projective_dynamics;

int main() {
    try {
        physica::examples::cloth::projective_dynamics::Simulation simulation{};
        const physica::examples::cloth::projective_dynamics::Summary summary = simulation.run();

        std::println("algorithm=ProjectiveDynamicsBouaziz2014");
        std::println("membrane_energy=stiffness*uv_area/2*||F-P||_F^2 local_projection=exact-thin-polar");
        std::println("bending_energy=stiffness/2*(opposite-distance-rest-length)^2 local_projection=rest_length*direction");
        std::println("global_system=free-only-upper-CSR solver=cuDSS-SPD deterministic_solves=3x-single-RHS");
        std::println("global_iterations={} membrane_stiffness={:.6f} bending_stiffness={:.6f}", simulation.global_iteration_count, simulation.membrane_stiffness, simulation.bending_stiffness);
        std::println("mesh={}x{} width={:.6f} height={:.6f} anchors=top-corners fixed_particles=({}, {}) gravity=(0,{:.6f},0) mass={:.6f}", simulation.rows, simulation.columns, simulation.width, simulation.height, simulation.fixed_particles[0], simulation.fixed_particles[1], simulation.gravity_y, simulation.mass);
        std::println("initial_perturbation_z={:.6f} profile=sin(row)*sin(2*column)", simulation.initial_perturbation);
        std::println("frames={} dt={:.9f} physical_time={:.6f}", summary.frames, simulation.time_step, summary.physical_time);
        std::println("maximum_material_stretch_ratio={:.9f}", summary.maximum_material_stretch_ratio);
        std::println("maximum_deformation_metric={:.9e}", summary.maximum_deformation_metric);
        std::println("maximum_bending_distance_error={:.9e}", summary.maximum_bending_distance_error);
        std::println("maximum_displacement={:.9e}", summary.maximum_displacement);
        std::println("maximum_fixed_position_error={:.9e}", summary.maximum_fixed_position_error);
        std::println("probe_particle={} probe_position=({:.9f},{:.9f},{:.9f})", simulation.probe_particle, summary.probe_position.x, summary.probe_position.y, summary.probe_position.z);
        std::println("probe_velocity=({:.9f},{:.9f},{:.9f})", summary.probe_velocity.x, summary.probe_velocity.y, summary.probe_velocity.z);
        return 0;
    } catch (const std::exception& error) {
        std::println(std::cerr, "cloth Projective Dynamics example failed: {}", error.what());
        return 1;
    }
}
