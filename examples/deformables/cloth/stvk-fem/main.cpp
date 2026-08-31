#include <physica/cuda.h>

import std;
import physica.example.deformables.cloth.stvk_fem;

int main() {
    try {
        physica::examples::cloth::stvk_fem::Simulation simulation{};
        const physica::examples::cloth::stvk_fem::Summary summary = simulation.run();

        std::println("algorithm=SaintVenantKirchhoffCST");
        std::println("integration=backward-Euler incremental-potential Newton=exact-gradient-and-Hessian globalization=GPU-batched-Armijo solver=GPU-block-PCG");
        std::println("material=plane-stress-StVK energy=thickness*rest-UV-area*(mu*tr(E^2)+lambda/2*tr(E)^2) strain=Green-Lagrange");
        std::println("regularization=free-free-Gershgorin-shift raw-Hessian=preserved preconditioner=exact-3x3-block-Jacobi");
        std::println("mesh={}x{} width={:.6f} height={:.6f} anchors=top-corners fixed_particles=({}, {}) gravity=(0,{:.6f},0) mass={:.6f}", simulation.rows, simulation.columns, simulation.width, simulation.height, simulation.fixed_particles[0], simulation.fixed_particles[1], simulation.gravity_y, simulation.mass);
        std::println("young_modulus={:.3e} poisson_ratio={:.6f} thickness={:.7f}", simulation.young_modulus, simulation.poisson_ratio, simulation.thickness);
        std::println("newton_iterations={} pcg_iterations={} line_search_candidates={} contraction=0.5", simulation.newton_iteration_count, simulation.pcg_iteration_count, simulation.line_search_candidate_count);
        std::println("initial_perturbation_z={:.6f} initial_velocity_z={:.6f}", simulation.initial_perturbation, simulation.initial_velocity);
        std::println("frames={} dt={:.9f} physical_time={:.6f}", summary.frames, simulation.time_step, summary.physical_time);
        std::println("maximum_absolute_principal_green_strain={:.9e}", summary.maximum_absolute_green_strain);
        std::println("total_elastic_energy={:.9e}", summary.total_elastic_energy);
        std::println("final_regularization_shift={:.9e} final_accepted_step={:.9f} final_candidate={}", summary.regularization_shift, summary.accepted_step_size, summary.accepted_line_search_candidate);
        std::println("maximum_fixed_position_error={:.9e}", summary.maximum_fixed_position_error);
        std::println("maximum_position_magnitude={:.9f} maximum_velocity_magnitude={:.9f}", summary.maximum_position_magnitude, summary.maximum_velocity_magnitude);
        std::println("probe_particle={} probe_displacement={:.9e} probe_position=({:.9f},{:.9f},{:.9f})", simulation.probe_particle, summary.probe_displacement, summary.probe_position.x, summary.probe_position.y, summary.probe_position.z);
        std::println("probe_velocity=({:.9f},{:.9f},{:.9f})", summary.probe_velocity.x, summary.probe_velocity.y, summary.probe_velocity.z);
        return 0;
    } catch (const std::exception& error) {
        std::println(std::cerr, "cloth StVK FEM example failed: {}", error.what());
        return 1;
    }
}
